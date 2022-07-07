"""
Computer Speed Test
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 3/28/2022
Updated: 3/28/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description: In 1952 it took approximately 6 hours to perform 20,000,000
multiplications. In 2022, how long does it take on a consumer-grade
computer?
"""
from datetime import datetime

# Get the starting time.
start = datetime.now()

# Preform 20,000,000 multiplications.
for i in range(20_000_000):
    product = 7 * 60

# Get the ending time and print the time elapsed.
end = datetime.now()
print('Time to perform 20,000,000 multiplications:', end - start)
