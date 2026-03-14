print("Challenge 35:")
name = input("Type in your Name: ")
for a in range(0,3):
    print(name)

print("Challenge 36:")
name0 = input("Type in your Name: ")
number = int(input("Enter a Number: "))
for b in range(0,number):
    print(name0)

print("Challenge 37:")
name1 = input("Enter your Name: ")
for c in name1:
    print(c)

print("Challenge 38:")
num = int(input("Enter a Number: "))
name2 = input("Enter your Name: ")
for d in range(0,num):
    for e in name2:
        print(e)

print("Challenge 39:")
num0 = int(input("Enter a Number between 1 and 12: "))
for f in range(1,13):
    answer = f * num0
    print(f,"x",num0,"=",answer)

print("Challenge 40:")
num1 = int(input("Enter a Number below 50: "))
for g in range(50,num1 - 1,-1):
    print(g)

print("Challenge 41:")
name3 = input("Enter your Name: ")
num2 = int(input("Enter a Number: "))
if num2 < 10:
    for h in range(0,num2):
        print(name3)
else:
    for i in range(0,3):
        print("Too High")

print("Challenge 42:")
total = 0
for j in range(0,5):
    num3 = int(input("Enter a Number: "))
    answer0 = input("Do you want this number included? (y/n)    ")
    if answer0 == "y":
        total = total + num3
print(total)

print("Challenge 43:")
direction = input("Do you want to count up or down? (u/d)    ")
if direction == "u":
    num4 = int(input("What is the Top Number? "))
    for k in range(1,num4 + 1):
        print(k)
elif direction == "d":
    num5 = int(input("Enter a Number below 20: "))
    for l in range(20,num5 - 1,-1):
        print(l)
else:
    print("I don't understand.")

print("Challenge 44:")
num6 = int(input("How many friends do you want to invite to the Party? "))
if num6 < 10:
    for m in range(0,num6):
        name4 = input("Enter a Name: ")
        print(name4,"has been invited.")
else:
    print("Too many People.")

print("Challenge 45:")
total0 = 0
while total0 <= 50:
    num7 = int(input("Enter a Number: "))
    total0 = total0 + num7
    print("The total is",total0)

print("Challenge 46:")
num8 = 0
while num8 <= 5:
    num8 = int(input("Enter a Number: "))
print("The Last Number you entered was a",num8)

print("Challenge 47:")
num9 = int(input("Enter a Number: "))
total1 = num9
again = "y"
while again == "y":
    num10 = int(input("Enter another Number: "))
    total1 = total1 + num10
    again = input("Do you want to add another Number? (y/n)     ")
print("The Total is",total1)

print("Challenge 48:")
again0 = "y"
count = 0
while again0 == "y":
    name5 = input("Enter a Name of somebody, that you want to invite to your Party: ")
    print(name5,"has now been invited.")
    count = count + 1
    again0 = input("Do you want to invite somebody else? (y/n)     ")
print("You have",count,"people coming to your party.")

print("Challenge 49:")
compnum = 50
guess = int(input("Can you guess the Number, I am thinking of? "))
count0 = 1
while guess != compnum:
    if guess < compnum:
        print("Too Low")
    else:
        print("Too High")
    count0 = count0 + 1
    guess = int(input("Have another Guess: "))
print("Well Done! you took",count0,"attempts")

print("Challenge 50:")
num11 = int(input("Enter a Number between 10 and 20: "))
while num11 < 10 or num11 > 20:
    if num11 < 10:
        print("Too Low")
    else:
        print("Too High")
    num11 = int(input("Try Again"))
print("Thank You")

print("Challenge 51:")
num12 = 10
while num12 > 0:
    print("There are",num12,"green bottles hanging on the wall.")
    print("If 1 Green Bottle should accidentally fall.")
    num12 = num12 - 1
    answer1 = int(input("How many Green Bottles will be hanging on the wall? "))
    if answer1 == num12:
        print("There will be",num12,"Green Bottles hanging on the wall.")
    else:
        while answer1 != num12:
            answer1 = int(input("No, Try Again"))
print("There are no more green bottles hanging on the wall.")

print("Challenge 52:")
import random
num13 = random.randint(1,100)
print(num13)

print("Challenge 53:")
import random
fruit = random.choice(['Apple','Orange','Grape','Banana','Strawberry'])
print(fruit)

print("Challenge 54:")
import random
coin = random.choice(['h','t'])
guess0 = input("Enter (h)eads or (t)ails: ")
if guess0 == coin:
    print("You Win")
else:
    print("Bad Luck")
if coin == "h":
    print("It was Heads")
else:
    print("It was Tails")

print("Challenge 55:")
import random
num14 = random.randint(1,5)
guess1 = int(input("Enter a Number: "))
if guess1 == num14:
    print("Well Done")
elif guess1 > num14:
    print("Too High")
    guess1 = int(input("Guess Again: "))
    if guess1 == num14:
        print("Correct")
    else:
        print("You Lose")
elif guess1 < num14:
    print("Too Low")
    guess1 = int(input("Guess Again: "))
    if guess1 == num14:
        print("Correct")
    else:
        print("You Lose")

print("Challenge 56:")
import random
num15 = random.randint(1,10)
correct = False
while correct == False:
    guess2 = int(input("Enter a Number: "))
    if guess2 == num15:
        correct = True

print("Challenge 57:")
import random
num16 = random.randint(1,10)
correct0 = False
while correct0 == False:
    guess3 = int(input("Enter a Number: "))
    if guess3 == num16:
        correct0 = True
    elif guess3 > num16:
        print("Too High")
    else:
        print("Too Low")

print("Challenge 58:")
import random
score = 0
for x in range(1,6):
    num23 = random.randint(1,50)
    num20 = random.randint(1, 50)
    correct3 = num23 + num20
    print(num23,"+",num20,"=")
    ans = int(input("Your Answer: "))
    if ans == correct3:
        score = score + 1
print("You Scores",score,"out of 5")

print("Challenge 59:")
import random
color = random.choice(['red','blue','green','white','pink'])
print("Select from Red, Blue, Green, White, or Pink:")
tryagain = True
while tryagain == True:
    theirchoice = input("Enter a Color: ")
    theirchoice = theirchoice.lower()
    if color == theirchoice:
        print("Well Done")
        tryagain = False
    else:
        if color == "red":
            print("I bet you are seeing RED right now")
        elif color == "blue":
            print("Don't feel BLUE")
        elif color == "green":
            print("I bet you are GREEN with envy right now")
        elif color == "white":
            print("Are you WHITE as a sheet, as you didn't guess correctly")
        elif color == "pink":
            print("Shame you are not feeling in the PINK, as you got it wrong")
        else:
            print("Okay")