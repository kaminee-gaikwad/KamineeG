# E-shop assignment
import getpass


def invalid_name(lists, element_name):
    global name_dic
    for dic in lists:
        try:
            if (dic['user_name']) == element_name:
                return True
        except:
            if (dic['product_name']) == element_name:
                name_dic=dic
                return True
    return False


def print_elements_name(lists):
    for dic in lists:
        try:
            print('\t' + dic['user_name'])
        except:
            print('\t' + dic['product_name'])


def add_user():
    user_name = input('Enter the user name(b: return menu)')
    if user_name == 'b':
        return admin_menu()

    while invalid_name(users, user_name):
        user_name = input("user already exists, enter new user name: ")
    password = getpass.getpass('Enter the password')
    users.append({
        'user_id': len(users),
        'user_name': user_name,
        'password': password,
        'products': []
    })
    admin_menu()


def remove_user():
    print('This is users list:')
    print_elements_name(users)
    user_name=input('enter the usernsme that you want to ramove, (b: return to menu)')
    if user_name == 'b':
        return admin_menu()
    remove_id=None
    for user in users:
        if user['user_name']==user_name:
            remove_id=user['user_id']
            print(user)
            break
    if remove_id is None:
         print('There no such user name')
    else:
        answer = input('Are you sure to remove user'+str(user_name)+'y:yes/n:no')
        if answer == 'y':
            del users[remove_id]
        admin_menu()


def add_product():
    product_name = input('Enter the product name(b: return menu)')
    if product_name == 'b':
        return admin_menu()

    if invalid_name(products, product_name):
        answer=input('product name alraedy exists, Do you want to increase quantity or choose' 
                     '\n(i:increase the quanity /c: chose an another name:').lower()
        if answer == 'c':
            return admin_menu()
        elif answer == 'i':
            quantity_to_add=int(input('how much units to add?:'))
            name_dic['product_quantity'] += quantity_to_add
            return
        else:
            print('wrong answer!')
            add_product()

    product_name = input("Enter the product name: ")
    product_price = int(input('Enter price of product'))
    product_quantity = int(input('Enter quantity of product'))

    products.append({
        'product_id': len(products),
        'product_name': product_name,
        'product_price': product_price,
        'product_quantity': product_quantity
    })
    admin_menu()


def remove_product():
    print('This is products list:')
    print_elements_name(products)
    product_name = input('enter the product name that you want to ramove, (b: return to menu)')
    if product_name == 'b':
        return admin_menu()
    remove_id = None
    for product in products:
        if product['user_name'] == product_name:
            remove_id = product['user_id']
            print(product)
            break
    if remove_id is None:
        print('***There no such product name***')
    else:
        answer = input('Are you sure to remove product' + str(product_name) + 'y:yes/n:no')
        if answer == 'y':
            del products[remove_id]
        admin_menu()


def check_lists(list):
    i=1
    for dic in list:
        print('('+str(i)+')', dic)
        i += 1


def check_users():
    check_lists(users)
    admin_menu()


def check_products():
    check_lists(products)
    admin_menu()


def finish():
    print("Thanks for choosing us!")
    quit()


def admin_menu():
    task_number = input('---------------------.\n' '(1) Add the user .\n (2) Remove the user .\n '
                        '(3) Add the Product .\n (4) Remove the products .\n '
                        '(5) Check Users.\n (6) Check Products.\n (7) quit, please choose an operation')
    while int(task_number) not in range(1, 8):
        task_number = input('Incorrect operation, Enter correct number:')
    return {
        '1': add_user,
        '2': remove_user,
        '3': add_product,
        '4': remove_product,
        '5': check_users,
        '6': check_products,
        '7': quit
    }[task_number]()


products = [
    {
        'product_id': 0,
        'product_name': 'Reebok shoes',
        'product_price($)': 300,
        'product_quantity': 35
    },
    {
        'product_id': 1,
        'product_name': 'Nike T-shirt',
        'product_price($)': 400,
        'product_quantity': 30
    },
    {
        'product_id': 2,
        'product_name': 'Hat',
        'product_price($)': 100,
        'product_quantity': 20
    },
    {
        'product_id': 3,
        'product_name': 'Cycle',
        'product_price($)': 700,
        'product_quantity': 25
    }
]

users = [
    {
        'user_name': 'admin',
        'password': 'admin1234'
    },
    {
        'user_id': 0,
        'user_name': 'John',
        'password': 'john1234',
        'products': [products[0], products[1]]

    },
    {
        'user_id': 1,
        'user_name': 'Maya',
        'password': 'maya1234',
        'products': [products[1], products[2]]
    },
    {
        'user_id': 2,
        'user_name': 'Gita',
        'password': 'gita1234',
        'products': [products[1], products[3]]
    },
    {
        'user_id': 3,
        'user_name': 'Ram',
        'password': 'ram1234',
        'products': [products[0], products[2]]
    }
    # to add new user we can use user_id as length of users
]


def user_menu():
    answer = input('\t(1)Shopping\n'
                   '\t(2)check your cart\n'
                   '\t(3)Remove a product from the cart\n'
                   '\t(4)Finish\n'
                   'Please choose an operation: ')
    while int(answer) not in range(1, 5):
        answer = print('Incorrect option chose correct option: ')
    return {
        '1': shopping,
        '2': check_cart,
        '3': remove_product_from_cart,
        '4': finish
    }[answer]()


def shopping():
    i = 0
    print('Products list:')
    for product in products:
        print('(' + str(i) + ')', product['product_name'], str(product['product_price($)']) + '$',
              str(product['product_quantity']) + 'u')
        i += 1
    product_number = int(input('Enter the product number that you want to add to your cart'
                               '(-1: return to the menu): '))
    if product_number == -1:
        return user_menu()
    if product_number in range(0, len(products)):
        users[logged_user]['products'].append(products[product_number])
        products[product_number]['product_quantity'] -= 1
    else:
        print("The number choosed doesn't exist in the products list")
        shopping()
    user_menu()


#Add products to cart



def check_cart():
    try:
        sum = 0
        i = 0
        for product in users[logged_user]['products']:
            print('(' + str(i) + ')', product['product_name'], '\t', str(product['product_price']) + '$')
            sum += product['product_price']
            i += 1
        print('\n\tThe sum is:', str(sum) + '$', end='\n___________________________________________________________\n')
    except:
        print('Your cart is empty', end='\n___________________________________________________________\n')
    user_menu()


def remove_product_from_cart():
    try:
        i = 0
        for product in users[logged_user]['products']:
            print('(' + str(i) + ')', product['product_name'], '\t', product['product_price'], "$")
            i += 1
    except:
        print('You cart is empty', end='\n___________________________________________________________\n')
        return user_menu()
    product_to_remove = int(input('Please enter the number of the product that you want to remove: '))
    if product_to_remove in range(0, len(users[logged_user]['products'])):
        answer = input('Are you sure that you want to remove '
                       + users[logged_user]['products'][product_to_remove]['product_name'].strip()
                       + '(y:Yes/n:No): ').lower()
        if answer == 'y':
            users[logged_user]['products'][product_to_remove]['product_quantity'] += 1
            del users[logged_user]['products'][product_to_remove]

        else:
            return user_menu()
    else:
        print("Product number choosed doesn't exist in your cart")
    return user_menu()


name_dic = None
tries=5
logged_user = None
answer = input('(1) Register \n(2) Sign in \nPlease choose an option: ')
while int(answer) not in range(1, 3):
    answer = input('(1) Register \n(2) Sign in \nPlease chose an option: ')
if answer == '1':
    add_user()
elif answer == '2':
    while tries > 0:
        user_name = input('Please enter your name: ')
        user_password = getpass.getpass('Please enter your password: ')
        for user in users:
            if user_name == user['user_name'] and user_password == user['password']:
                print('Hello', user_name, '!')
                logged_user = user['user_id']
                tries = -1
                break
        if tries != -1:
            print('The information entered are incorrect')  # The information entered by the user are false
            tries -= 1
        if tries == 0:
            print('You have passed 5 tries!')
            quit()
if user_name == 'admin':
    admin_previliges = True
    admin_menu()
else:
    user_menu()