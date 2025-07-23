import numpy as np
import random

class FailureSimulator:
    def __init__(self, failure_prob=0.05, failure_type="disconnect"):
        self.failure_prob = failure_prob
        self.failure_type = failure_type
        self.active_failures = {}
        
    def simulate_failure(self, agent_id, action, step):
        if agent_id not in self.active_failures:
            if random.random() < self.failure_prob:
                duration = random.randint(5, 20)
                self.active_failures[agent_id] = {
                    'start_step': step,
                    'duration': duration,
                    'type': self.failure_type
                }
        
        if agent_id in self.active_failures:
            failure = self.active_failures[agent_id]
            elapsed = step - failure['start_step']
            
            if elapsed < failure['duration']:
                if failure['type'] == "disconnect":
                    return np.zeros_like(action)
                elif failure['type'] == "noise":
                    return action + np.random.normal(0, 0.5, size=action.shape)
                elif failure['type'] == "random":
                    if isinstance(action, int):
                        return random.randint(0, action)
                    return np.random.uniform(-1, 1, size=action.shape)
            else:
                del self.active_failures[agent_id]
                
        return action
    
    def is_agent_failed(self, agent_id):
        return agent_id in self.active_failures