
def split(n, count):
    count += 1
    if n == 1:
        return 1
    for i in range(n):
        count += split(n-i-1, count)
    return count
count = 0
print(split(4, count))