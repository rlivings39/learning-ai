# bounce.py
#
# Exercise 1.5
def main():
    height = 100 # meters
    for bounce_number in range(10):
        height = round(height*(3/5), 4)
        print(bounce_number+1, height)

if __name__ == "__main__":
    main()
