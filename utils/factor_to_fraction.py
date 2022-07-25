from math import floor
 
# Recursive function to
# return GCD of a and b
def gcd(a, b):
     
    if (a == 0):
        return b
    elif (b == 0):
        return a
    if (a < b):
        return gcd(a, b % a)
    else:
        return gcd(b, a % b)
 
    # Function to convert decimal to fraction

def FactorToFraction(number):
     
    # Fetch integral value of the decimal
    intVal = floor(number)
 
    # Fetch fractional part of the decimal
    fVal = number - intVal
 
    # Consider precision value to
    # convert fractional part to
    # integral equivalent
    pVal = 1000000000
 
    # Calculate GCD of integral
    # equivalent of fractional
    # part and precision value
    gcdVal = gcd(round(fVal * pVal), pVal)
 
    # Calculate num and deno
    num= round(fVal * pVal) // gcdVal
    deno = pVal // gcdVal
 
    # Print the fraction
    return ((intVal * deno) + num), deno