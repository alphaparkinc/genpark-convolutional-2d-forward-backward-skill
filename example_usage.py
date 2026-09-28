from client import Conv2D

def main():
    img = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    k = [[1.0, 0.0], [0.0, 1.0]]
    res = Conv2D.forward(img, k, stride=1, padding=0)
    print("Conv2D Result:")
    for row in res:
        print(" ", row)

if __name__ == "__main__":
    main()
