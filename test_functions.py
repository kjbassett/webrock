from webrock.decorator import plugin

# 1. Simple function
@plugin()
def greet(name):
    return f"Hello, {name}!"


# 2. Function with default argument
@plugin()
def power(base, exponent=2):
    return base ** exponent


# 3. Function with conditional logic
@plugin()
def is_even(n):
    return n % 2 == 0


# 4. Function working with lists
@plugin()
def average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)


# 5. Function returning multiple values
@plugin()
def min_max(values):
    return min(values), max(values)


# 6. Function using *args
def add_all(*nums):
    return sum(nums)


# 7. Function with keyword arguments
def build_url(base, **params):
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{base}?{query}" if query else base


# 8. Recursive function
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

import asyncio
from webrock.app import create_app


def main():
    app = asyncio.run(create_app())
    app.run(host="0.0.0.0", port=8000, single_process=True)


if __name__ == "__main__":
    main()