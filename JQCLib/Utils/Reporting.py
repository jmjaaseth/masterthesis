###########################################
#      Code By Jørgen Mjaaseth, 2023      #
###########################################

import os
from colorama import Fore, Style
 
# define clear function
def clear():

    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')

def reportProgress(models, waiting, name):

    #Clear the screen
    clear()

    LEFT = "     "
    WIDTH = 58 + len(name) + len(LEFT)

    def header(a,b):
        s = LEFT + a
        for i in range(WIDTH-len(s)-1):
            s+="─"
        print(s+b)  
    
    def txt(text):
        s = LEFT + "│" + text
        print(s.ljust(WIDTH-1) + "│")

    print(Fore.GREEN)
    header("┌","┐")
    txt("  ╔═══╗   TRAINING QUANTUM CIRCUITS!")
    txt("  ╚═╦═╝   ──────────────────────────")
    txt("          THANKS FOR CONTRIBUTING TO MY THESIS, " + name.upper())

    mtext = ""
    if(models > 0):        
        mtext = "          YOU HAVE TRAINED " + str(models) + " MODEL" + ("S" if models > 1 else "") + "!!!"
    txt(mtext)

    mtext = ""
    if(waiting > 0):        
        mtext = "          MODELS WAITING TO BE UPLOADED: " + str(waiting)
    txt(mtext)

    if(waiting == 0):
        txt("")
    else:
        txt("          CHECK YOUR INTERNET CONNECTION OR FIREWALL!!!")
    header("└","┘")

    print(Style.RESET_ALL)