

import numpy as np
from collections import deque

def initialize_stack(maxlen=4):
    return deque(maxlen=maxlen)

def append_to_stack(stack, array):
    stack.append(array)
    return stack

def get_current_stack(stack):
    # If the stack isn't full yet, this will pad it with zeros
    while len(stack) < 4:
        stack.appendleft(np.zeros((1, 1, 84, 84)))
    return np.concatenate(list(stack), axis=1)

# Example usage:

# Initializing the stack
stack = initialize_stack()

# Let's append some sample arrays to the stack:
for _ in range(5):  # appending 5 arrays, so the first one should be removed automatically
    #sample_array = np.random.randn(1, 1, 84, 84)
    sample_array = np.ones((1, 1, 84, 84))
    stack = append_to_stack(stack, sample_array)
    print(get_current_stack(stack).shape)  # This should always print (1, 4, 84, 84) after the 4th append
    print(stack[0])

# The shape of the resulting stack is:
print(get_current_stack(stack).shape)  # Expected: (1, 4, 84, 84)

exit()



# Create a sample 4D matrix with shape (2, 3, 4, 5)
tensor1 = np.random.rand(2, 3, 4, 5)
tensor2 = np.random.rand(2, 3, 4, 5)


# Push tensor1 onto the stack
stack.append(tensor1)

while len(stack) < 4:
    # For demonstration purposes, let's append tensor2 to tensor1 along axis=1 (second dimension)
    combined_tensor = np.concatenate((stack[-1], tensor2), axis=1)  # this results in a tensor with shape (2, 6, 4, 5)

    # Now, let's push the combined_tensor onto the stack
    stack.append(combined_tensor)

    print(stack[-1].shape)  # Expected output: (2, 6, 4, 5)
