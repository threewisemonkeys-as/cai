import os
import time
from datasets import load_dataset
from r2egym.agenthub.runtime.docker import DockerRuntime

# Since direct Docker works, the issue is in DockerRuntime implementation
# Let's try different approaches

def test_dockerruntime_variations():
    """Test different DockerRuntime configurations"""
    
    os.environ["DOCKER_HOST"] = "ssh://dockerd-remote"
    
    # Load dataset
    ds = load_dataset("R2E-Gym/R2E-Gym-Subset", split="train")
    sample = ds[0]
    
    print(f"Testing with image: {sample.get('docker_image', 'unknown')}")
    
    # Test 1: Different shell command
    variations = [
        {
            'name': 'Original (bash -l)',
            'command': ["/bin/bash", "-l"]
        },
        {
            'name': 'Simple bash',
            'command': ["/bin/bash"]
        },
        {
            'name': 'Simple sh',
            'command': ["/bin/sh"]
        },
        {
            'name': 'Explicit sh exec',
            'command': ["sh", "-c", "exec sh"]
        },
        {
            'name': 'Cat for stdin',
            'command': ["cat"]  # Sometimes used for stdin-based execution
        }
    ]
    
    for i, variation in enumerate(variations, 1):
        print(f"\n{i}. Testing {variation['name']}...")
        remote = None
        
        try:
            # Set potential Docker SDK environment variables
            old_timeout = os.environ.get('DOCKER_CLIENT_TIMEOUT')
            os.environ['DOCKER_CLIENT_TIMEOUT'] = '300'  # 5 minutes
            
            remote = DockerRuntime(
                ds=sample,
                command=variation['command'],
                backend="docker",
            )
            print(f"   ✓ DockerRuntime created successfully")
            
            # Try the command that was failing
            print(f"   Testing 'ls /testbed'...")
            result = remote.run("ls /testbed")
            
            if "BrokenPipeError" in str(result):
                print(f"   ✗ BrokenPipeError: {result}")
            elif "Error:" in str(result):
                print(f"   ✗ Other error: {result}")
            else:
                print(f"   ✓ SUCCESS: {result}")
                print(f"   ^^ This variation works! Use: {variation['command']}")
                
                # Test one more command to be sure
                result2 = remote.run("echo 'test successful'")
                print(f"   ✓ Second test: {result2}")
                break
                
        except Exception as e:
            print(f"   ✗ Exception during creation: {e}")
            
        finally:
            if remote:
                try:
                    remote.close()
                except:
                    pass
            
            # Restore timeout
            if old_timeout:
                os.environ['DOCKER_CLIENT_TIMEOUT'] = old_timeout
            else:
                os.environ.pop('DOCKER_CLIENT_TIMEOUT', None)
        
        # Small delay between tests
        time.sleep(2)


def test_environment_variables():
    """Test with different Docker SDK environment variables"""
    
    print("\n" + "="*50)
    print("Testing Environment Variable Solutions")
    print("="*50)
    
    env_variations = [
        {
            'name': 'Default Docker API',
            'vars': {}
        },
        {
            'name': 'Older API version',
            'vars': {
                'DOCKER_API_VERSION': '1.39',
                'COMPOSE_API_VERSION': '1.39'
            }
        },
        {
            'name': 'Disable experimental features',
            'vars': {
                'DOCKER_CLI_EXPERIMENTAL': 'disabled',
                'COMPOSE_DOCKER_CLI_BUILD': '0'
            }
        },
        {
            'name': 'Increase timeouts',
            'vars': {
                'DOCKER_CLIENT_TIMEOUT': '300',
                'COMPOSE_HTTP_TIMEOUT': '300'
            }
        }
    ]
    
    # Load dataset once
    ds = load_dataset("R2E-Gym/R2E-Gym-Subset", split="train")
    sample = ds[0]
    
    for i, env_test in enumerate(env_variations, 1):
        print(f"\n{i}. Testing {env_test['name']}...")
        
        # Store original values
        original_vars = {}
        
        try:
            # Set test environment variables
            for key, value in env_test['vars'].items():
                original_vars[key] = os.environ.get(key)
                os.environ[key] = value
                print(f"   Set {key}={value}")
            
            # Test DockerRuntime
            remote = None
            try:
                remote = DockerRuntime(
                    ds=sample,
                    command=["/bin/bash", "-l"],  # Use original command
                    backend="docker",
                )
                
                result = remote.run("ls /testbed")
                
                if "BrokenPipeError" not in str(result) and "Error:" not in str(result):
                    print(f"   ✓ SUCCESS with environment vars: {env_test['vars']}")
                    print(f"   Result: {result}")
                    break
                else:
                    print(f"   ✗ Still failing: {result}")
                    
            except Exception as e:
                print(f"   ✗ Exception: {e}")
            finally:
                if remote:
                    remote.close()
                    
        finally:
            # Restore original environment variables
            for key, original_value in original_vars.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value


def main():
    print("Targeted DockerRuntime Fix Testing")
    print("Since direct Docker commands work, the issue is in DockerRuntime implementation")
    print("="*80)
    
    os.environ["DOCKER_HOST"] = "ssh://dockerd-remote"
    
    test_dockerruntime_variations()
    test_environment_variables()
    
    print("\n" + "="*80)
    print("If none of these work, the issue is likely in the DockerRuntime source code")
    print("itself, specifically how it uses the Python Docker SDK vs CLI commands.")


if __name__ == "__main__":
    main()