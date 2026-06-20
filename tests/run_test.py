import sys
import os
import types

# Ensure coverage module exists with standard types bypass to avoid numba import crash
class DynamicMock:
    def __getattr__(self, name):
        return object

coverage_mock = DynamicMock()
coverage_mock.types = DynamicMock()
sys.modules['coverage'] = coverage_mock



# Add virtualenv path and workspace root
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_dir, ".venv/lib/python3.10/site-packages"))
sys.path.insert(1, workspace_dir)

from tests.test_sensor_suite import test_tf_manager, test_sensor_suite_init, test_sensor_suite_scheduling
from tests.test_refactoring import test_planner_factory, test_unsupported_planner, test_simulator_factory

if __name__ == "__main__":
    print("==========================================")
    print("Starting Sensor Suite & Refactoring Tests...")
    print("==========================================")
    
    try:
        print("1. Running test_tf_manager...")
        test_tf_manager()
        print("   -> Passed!")
        
        print("2. Running test_sensor_suite_init...")
        test_sensor_suite_init()
        print("   -> Passed!")
        
        print("3. Running test_sensor_suite_scheduling...")
        test_sensor_suite_scheduling()
        print("   -> Passed!")
        
        print("4. Running test_planner_factory...")
        test_planner_factory()
        print("   -> Passed!")
        
        print("5. Running test_unsupported_planner...")
        test_unsupported_planner()
        print("   -> Passed!")
        
        print("6. Running test_simulator_factory...")
        test_simulator_factory()
        print("   -> Passed!")
        
        print("==========================================")
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("==========================================")
    except Exception as e:
        print(f"\n[Test Failure] Error encountered during tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

