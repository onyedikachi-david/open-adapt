"""Module for testing the recording module."""

import multiprocessing
import time
import os
import signal
import pytest
import psutil
from unittest.mock import patch
from openadapt import record, playback, utils, video
from openadapt.config import config
from openadapt.db import crud
from openadapt.models import Recording, ActionEvent
from loguru import logger

RECORD_STARTED_TIMEOUT = 360  # Increased timeout to 6 minutes
MAX_START_RETRIES = 3  # Maximum number of retries for starting recording
RETRY_DELAY = 5  # Delay between retries in seconds

# Mock window state for CI environment
MOCK_WINDOW_STATE = {
    'app': 'Terminal',
    'title': 'CI Test Window',
    'x': 0,
    'y': 0,
    'width': 800,
    'height': 600,
    'pid': os.getpid()  # Use current process PID
}

def is_ci_environment():
    """Check if we're running in a CI environment."""
    return os.environ.get('CI') == 'true'

def mock_get_active_window_state():
    """Mock window state for CI environment."""
    return MOCK_WINDOW_STATE

def is_process_running(pid):
    """Safely check if a process is running."""
    try:
        # First try sending signal 0 - doesn't actually send a signal
        os.kill(pid, 0)
        return True
    except OSError:
        return False
    except Exception:
        return False

def terminate_process_safe(process):
    """Safely terminate a process and ensure it's cleaned up."""
    if not process or not process.is_alive():
        return
    
    try:
        pid = process.pid
        process.terminate()
        process.join(timeout=2)  # Give it 2 seconds to terminate gracefully
        
        # If still alive, force kill
        if process.is_alive():
            logger.warning(f"Process {pid} didn't terminate gracefully, force killing...")
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # Process already gone
            process.join(timeout=1)
            
        # Final check for zombie process
        try:
            os.kill(pid, 0)
            logger.warning(f"Process {pid} might be a zombie, attempting cleanup")
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
        except OSError:
            pass  # Process is gone
    except Exception as e:
        logger.warning(f"Error while terminating process: {e}")

@pytest.fixture(autouse=True)
def setup_ci_mocks():
    """Setup mocks for CI environment."""
    if is_ci_environment():
        # Patch the window state function for CI
        with patch('openadapt.window.get_active_window_state', 
                  side_effect=mock_get_active_window_state):
            yield
    else:
        yield

@pytest.fixture
def setup_db():
    # Setup the database connection and return the session
    db_session = crud.get_new_session(read_and_write=True)
    yield db_session
    db_session.close()


def test_record_functionality():
    logger.info("Starting test_record_functionality")
    logger.info(f"Running in CI environment: {is_ci_environment()}")
    
    record_process = None
    parent_conn = None
    child_conn = None
    
    for attempt in range(MAX_START_RETRIES):
        logger.info(f"Starting recording attempt {attempt + 1}/{MAX_START_RETRIES}")
        
        # Clean up any existing connections
        if parent_conn:
            parent_conn.close()
        if child_conn:
            child_conn.close()
            
        # Set up multiprocessing communication
        parent_conn, child_conn = multiprocessing.Pipe()

        # Set up termination events
        terminate_processing = multiprocessing.Event()
        terminate_recording = multiprocessing.Event()

        # Start the recording process
        record_process = multiprocessing.Process(
            target=record.record,
            args=(
                "Test recording",
                terminate_processing,
                terminate_recording,
                child_conn,
                False,
            ),
        )
        
        try:
            record_process.start()
            pid = record_process.pid
            logger.info(f"Recording process started (PID: {pid})")

            # Wait for the 'record.started' signal
            start_time = time.time()
            signal_received = False
            
            while time.time() - start_time < RECORD_STARTED_TIMEOUT:
                if parent_conn.poll(1):  # 1 second timeout for poll
                    try:
                        message = parent_conn.recv()
                        logger.info(f"Received message: {message}")
                        if message["type"] == "record.started":
                            logger.info("Received 'record.started' signal")
                            signal_received = True
                            break
                    except EOFError:
                        logger.error("Connection closed unexpectedly")
                        break
                else:
                    logger.debug("No message received, continuing to wait...")
                    
                # Check if process is still alive using our safe function
                if not is_process_running(pid):
                    logger.error(f"Recording process (PID: {pid}) died unexpectedly")
                    break
                    
            if signal_received:
                break  # Successfully started recording
            
            logger.warning(f"Recording attempt {attempt + 1} failed")
            
            # Clean up failed attempt
            terminate_process_safe(record_process)
            
            if attempt < MAX_START_RETRIES - 1:
                logger.info(f"Waiting {RETRY_DELAY} seconds before next attempt")
                time.sleep(RETRY_DELAY)
            else:
                logger.error("All recording attempts failed")
                pytest.fail("Timed out waiting for 'record.started' signal after all retries")

        except Exception as e:
            logger.exception(f"An error occurred during recording attempt {attempt + 1}: {e}")
            terminate_process_safe(record_process)
            
            if attempt < MAX_START_RETRIES - 1:
                continue
            raise

    try:
        # Wait a short time to ensure some data is recorded
        record_time = 10 if is_ci_environment() else 5
        logger.info(f"Recording for {record_time} seconds")
        time.sleep(record_time)

        logger.info("Stopping the recording")
        terminate_processing.set()  # Signal the recording to stop

        # Wait for the recording to stop with a shorter timeout in CI
        stop_timeout = 30 if is_ci_environment() else RECORD_STARTED_TIMEOUT
        logger.info(f"Waiting for recording to stop (timeout: {stop_timeout}s)")
        terminate_recording.wait(timeout=stop_timeout)
        
        if not terminate_recording.is_set():
            logger.error("Recording did not stop within the expected time")
            # Force terminate if needed
            terminate_process_safe(record_process)
            pytest.fail("Recording did not stop within the expected time")

        logger.info("Recording stopped successfully")

        # Assert database state
        with crud.get_new_session(read_and_write=True) as session:
            recording = session.query(Recording).order_by(Recording.id.desc()).first()
            assert recording is not None, "No recording was created in the database"
            assert recording.task_description == "Test recording"
            logger.info("Database assertions passed")

        # Assert filesystem state
        video_path = video.get_video_file_path(recording.timestamp)
        if config.RECORD_VIDEO:
            assert os.path.exists(video_path), f"Video file not found at {video_path}"
            logger.info(f"Video file found at {video_path}")
        else:
            logger.info("Video recording is disabled in the configuration")

        performance_plot_path = utils.get_performance_plot_file_path(recording.timestamp)
        assert os.path.exists(performance_plot_path), f"Performance plot not found at {performance_plot_path}"
        logger.info(f"Performance plot found at {performance_plot_path}")

        # Assert that at least one action event was recorded
        with crud.get_new_session(read_and_write=True) as session:
            action_events = crud.get_action_events(session, recording)
            assert len(action_events) > 0, "No action events were recorded"
            logger.info(f"Number of action events recorded: {len(action_events)}")

        logger.info("All assertions passed")

    except Exception as e:
        logger.exception(f"An error occurred during the test: {e}")
        raise

    finally:
        # Clean up resources
        if record_process:
            logger.info("Cleaning up recording process")
            terminate_process_safe(record_process)
        if parent_conn:
            parent_conn.close()
        if child_conn:
            child_conn.close()
        logger.info("Test completed")


if __name__ == "__main__":
    pytest.main([__file__])
