print("Challenge 1:")
firstname = input("Please Enter your Name: ")
print("Hi!",firstname)
print("\n")

print("Challenge 2:")
firstname1 = input("Enter your First Name: ")
surname = input("Enter your Surname: ")
print("Hi!",firstname,surname)
print("\n")

print("Challenge 3:")
print("What do you call a bear with no teeth?\nA Gummy Bear!")
print("\n")

print("Challenge 4:")
num1 = float(input("Enter your 1st Number: "))
num2 = float(input("Enter your 2nd Number: "))
answer = num1 + num2
print("The Answer is:",answer)
print("\n")

print("Challenge 5:")
num01 = float(input("Enter your 1st Number: "))
num02 = float(input("Enter your 2nd Number: "))
num03 = float(input("Enter your 3rd Number: "))
answer1 = (num01 + num02) * num03
print("The Answer is:",answer1)
print("\n")

print("Challenge 6:")
startNum = int(input("Enter the number of slices of pizza you started with: "))
endNum = int(input("How many slices have you eaten? "))
slicesleft = startNum - endNum
print("You have ",slicesleft," slices remaining.")
print("\n")

print("Challenge 7:")
name = input("What is your Name? ")
age = int(input("How old are you? "))
newAge = age + 1
print(name,"next birthday you will be", newAge)
print("\n")

print("Challenge 8:")
bill = float(input("What is the total cost of the Bill? "))
people = int(input("How many people are there? "))
each = bill/people
print("Bill Division, Each Person: PKR", each)
print("\n")

print("Challenge 9:")
days = int(input("Enter the Number of Days: "))
hours = days * 24
minutes = hours * 60
seconds = minutes * 60
print("In ",days," days there are ",hours, " hours ",minutes," minutes ",seconds," seconds.")
print("\n")

print("Challenge 10:")
kilo = float(input("Enter the Number of Kilos: "))
pound = kilo * 2.204
print("That is",pound,"Pounds.")
print("\n")

print("Challenge 11:")
larger = float(input("Enter a Number over 100: "))
smaller = float(input("Enter a Number under 100: "))
answer = larger/smaller
print(smaller," goes into ",larger," ",answer," times")
print("\n")

print("Challenge 12:")
num001 = float(input("Enter 1st Number: "))
num002 = float(input("Enter 2nd Number: "))
if num001 > num002:
    print(num002, num001)
else:
    print(num001,num002)
print("\n")

print("Challenge 13:")
num = float(input("Enter a Value less than 20: "))
if num >= 20:
    print("Too High.")
else:
    print("Thank You.")
print("\n")

print("Challenge 14:")
num0 = float(input("Enter a Value between 10 and 20: "))
if num0 >= 10 and num0 <= 20:
    print("Thank You")
else:
    print("Incorrect Answer")
print("\n")

print("Challenge 15:")
color = input("Type in your Favourite Color: ")
if color == "Red" or color == "RED" or color == "red":
    print("I like Red too!")
else:
    print("I don't like that colour, I prefer Red.")
print("\n")

print("Challenge 16:")
raining = input("Is it Raining? ")
raining = str.lower(raining)
if raining == "yes":
    windy = input("Is it windy?")
    windy = str.lower(windy)
    if windy == "yes":
        print("It is too windy for an Umberalla.")
    else:
        print("Take an Umberalla.")
else:
    print("Enjoy your Day")
print("\n")

print("Challenge 17:")
age1 = int(input("What is your age?"))
if age >= 18:
    print("You can Vote")
elif age == 17:
    print("You can learn to drive.")
elif age == 16:
    print("You can buy a lottery ticket")
else:
    print("You can go Trick-or-Treating")
print("\n")

print("Challenge 18:")
num00 = float(input("Enter a Number: "))
if num00 < 10:
    print("Too Low")
elif num00 >= 10 and num <= 20:
    print("Correct")
else:
    print("Too High")
print("\n")

print("Challenge 19:")
num000 = input("Enter 1, 2, or 3: ")
if num000 == "1":
    print("Thank You")
elif num000 == "2":
    print("Well Done")
elif num000 == "3":
    print("Correct")
else:
    print("Error Message")
print("\n")

print("Challenge 20:")
name1 = input("Enter your First Name: ")
length = len(name1)
print(length)
print("\n")

print("Challenge 21:")
firstName = input("Enter your First Name: ")
surName = input("Enter your SurName: ")
name0 = firstName + " " + surName
length = len(name0)
print(name0)
print(length)
print("\n")

print("Challenge 22:")
first_name = input("Enter your First Name in LowerCase: ")
sur_name = input("Enter your Sur Name in LowerCase: ")
first_name = first_name.title()
sur_name = sur_name.title()
name11 = first_name + " " + sur_name
print(name11)
print("\n")

print("Challenge 23:")
phrase = input("Enter the 1st Line of a Nursery Rhyme: ")
length0 = len(phrase)
print("This has", length0, "letters in it")
start = int(input("Enter a Starting Point: "))
end = int(input("Enter a End Point: "))
part = (phrase[start:end])
print(part)
print("\n")

print("Challenge 24:")
word = input("Enter a Word: ")
word = word.upper()
print(word)
print("\n")

print("Challenge 25:")
name12 = input("Enter your 1st Name: ")
if len(name) < 5:
    _sur_name = input("Enter your Sur Name: ")
    name12 = name12 + _sur_name
    print(name.upper())
else:
    print(name.lower())
print("\n")

print("Challenge 26:")
word1 = input("Please Enter a Word: ")
first = word1[0]
length2 = len(word1)
rest = word1[1:length2]
if first != "a" and first != "e" and first != "i" and first != "o" and first != "u":
    newword = rest + first + "ay"
else:
    newword = word + "way"
print(newword.lower())
print("\n")

print("Challenge 27:")
num23 = float(input("Enter a Number with lots of Decimal Places: "))
print(num23)
print("\n")

print("Challenge 28:")
num13 = float(input("Enter a number with lots of decimal places: "))
answer = num * 2
print(answer)
print(round(answer,2))
print("\n")

print("Challenge 29:")
import math
num14 = float(input("Enter a Number over 500: "))
answer0 = math.sqrt(num14)
print(round(answer,2))
print("\n")

print("Challenge 30:")
import math
print(round(math.pi,5))
print("\n")

print("Challenge 31:")
import math
radius = float(input("Enter the Radius of the Circle: "))
area = math.pi * (radius ** 2)
print(area)
print("\n")

print("Challenge 32:")
import math
radius = float(input("Enter the Radius of the Circle: "))
depth = float(input("Enter Depth: "))
area = math.pi * (radius ** 2)
volume = area * depth
print(round(volume,3))
print("\n")

print("Challenge 33:")
num15 = float(input("Enter a Number: "))
num16 = float(input("Enter another Number: "))
ans = num1//num2
ans0 = num1%num2
print(num1," divided by ",num2," is ",ans," with ",ans0," remaining.")
print("\n")

print("Challenge 34:")
print("1. Square")
print("2. Triangle")
print()
menuselection = int(input("Enter a Number: "))
if menuselection == 1:
    side = float(input("Enter the Length of one Side: "))
    area = side * side
    print("The Area of your chosen shape is", area)
elif menuselection == 2:
    base = float(input("Enter the Length of the Base: "))
    height = float(input("Enter the Height of the Triangle: "))
    area = (base * height)/2
    print("The Area of your chosen Shape is", area)
else:
    print("Incorrect Option Selected")
print("\n")