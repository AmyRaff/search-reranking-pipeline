DIGITS = [str(a) for a in range(10)]


def add(a, b):
    return float(a) + float(b)

def subtract(a, b):
    return float(a) - float(b)

def multiply(a, b):
    return float(a) * float(b)

def divide(a, b):
    return float(a) / float(b)

def power(a, b):
    return float(a) ** float(b)

def is_digit(var):
    return var in DIGITS

OPERATIONS = {
    '+' : add,
    '-' : subtract,
    '*' : multiply,
    '/' : divide,
    '^' : power
}

def perform_operation(string, num1, num2):
    op = OPERATIONS.get(string, None)
    if op is not None:
        return op(num1, num2)
    else:
        print("No operation!")
        exit()
    
def get_number(varstr):
    s = ""
    for c in varstr:
        if not is_digit(c):
            break
        s += c
    return (int(s), len(s))

def eval_math_expr(expr):
    base_i = 0
    
    if expr[0] == '-':
        expr = '0' + expr
        
    digit_split = [is_digit(a) for a in expr]
    digits = [i for i in range(len(digit_split)) if digit_split[i] is True]
    operations = [expr[i] for i in range(len(digit_split)) if digit_split[i] is False]
    
    numbers = []
    base_i = 0
    for idx in digits: 
        if idx + 1 not in digits:
            numbers.append(expr[base_i:idx + 1]) 
            base_i = idx + 2 #
    
    for operation in operations:
        op = OPERATIONS.get(operation, None)
        if op is None:
            print("Invalid token '{}'. Please try again.\n".format(operation))
            exit()
    
    total = 0
    for i in range(len(numbers) - 1):
        if i == 0:
            total += perform_operation(operations[0], numbers[i], numbers[i+1])
        else:
            total = perform_operation(operations[0], total, numbers[i+1])
        operations.remove(operations[0])
        
    return float(total)


if __name__ == '__main__':
    
    assert eval_math_expr("5+5") == 10.0
    assert eval_math_expr("10-5") == 5.0
    assert eval_math_expr("5/5") == 1.0
    assert eval_math_expr("10*5") == 50.0
    assert eval_math_expr("5^2") == 25.0
    
    assert eval_math_expr("-5+3") == -2.0
    assert eval_math_expr("5+-3") == 2.0
    
    assert eval_math_expr("-5-3") == -8.0
    assert eval_math_expr("5--3") == 8.0
    
    assert eval_math_expr("-5*3") == -15.0 == eval_math_expr("5*-3")
    
    assert eval_math_expr("-30/3") == -10.0 == eval_math_expr("30/-3")
    
    assert eval_math_expr("-5^2") == 25.0

    expr = input('Enter your expression:')
    print(expr + ' = ')
    print(eval_math_expr(expr))
