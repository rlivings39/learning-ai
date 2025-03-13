def counter(val):
    def incr():
        nonlocal val
        val += 1
        return val

    def decr():
        nonlocal val
        val -= 1
        return val
    return incr, decr

def main():
    up, down = counter(0)
    print(up())
    print(up())
    print(up())
    print(down())
    print(down())
    print(down())

if __name__ == "__main__":
    main()
