from django.shortcuts import render
from .models import predict

def read_file(file_name):
    opened_file = open(file_name, 'r')
    lines_list = []
    for line in opened_file:
        line = line.split()
        lines_list.append(line)
    return lines_list

def home(request):
    if request.method == 'POST':
        # If the username or password is empty, show an error message
        name = request.POST.get('name')
        password = request.POST.get('password')
        if not name or not password:
            error_message = 'Please enter both username and password.'
            return render(request, 'index.html', {'error_message': error_message})

        # Proceed with authentication
        file_name = 'account.txt'
        account_list = read_file(file_name)

        for i in account_list:
            # Check the length of i before accessing its elements
            if len(i) >= 2 and i[0] == name and i[1] == password:
                return render(request, 'input.html')
        
        # If the loop completes without finding a matching account
        error_message = 'INVALID USERNAME OR PASSWORD'
        return render(request, 'index.html', {'error_message': error_message})

    return render(request, 'index.html')


class_names = ['INFP personality No borderline_personality_disorder',
               'INFJ with borderline_personality_disorder ',
               'INTP No borderline_personality_disorder',
               'INTJ No borderline_personality_disorder',
               'ENTP With borderline_personality_disorder']

def output(request):
    text = str(request.POST.get('text'))
    algo = request.POST.get('algo')
    out = predict(text, algo)
    return render(request, 'output.html', {'out': class_names[out]})


