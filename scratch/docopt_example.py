#!/usr/bin/env python

""" docopt_example.py -- This is an example script on the basics of how to use docopt. It takes in an input string, and saves it to a text file. It also prints it to the screen if verbose. 
Usage: docopt_example.py [-h] [-v] [-o SAVELOC] <input_string>

Arguments:
    input_string (string)

Options:
    -h, --help                              Show this screen
    -v, --verbose                           Show extra information [default: False]   
    -o SAVELOC, --out SAVELOC               Saved output as [default: ./temp.txt]

Examples:
python docopt_example.py -v lalal
python docopt_example.py lalal
python docopt_example.py -o './savehere.txt' lalal
python docopt_example.py -v -o './savehere.txt' lalal
"""

import docopt
import sys, os

__author__      = "Jielai Zhang"
__license__     = "MIT"
__version__     = "1.0.1"
__date__        = "2020-12-05"
__maintainer__  = "Jielai Zhang"
__email__       = "zhang.jielai@gmail.com"

##############################################################
####################### Main Function ########################
##############################################################

def docopt_example(input_string, saveloc='./temp.txt', verbose=False):

    # Save text file.
    f = open(saveloc,'w')
    f.write(input_string)
    f.close()
    print(f'Saved: {saveloc}')

    # If verbose, print input string.
    if verbose:
        print('VERBOSE: --- Printing input string: ---')
        print('VERBOSE: ',input_string)

    return saveloc

############################################################################
####################### BODY OF PROGRAM STARTS HERE ########################
############################################################################

if __name__ == "__main__":

    # Read in input arguments
    arguments       = docopt.docopt(__doc__)
    verbose         = arguments['--verbose']
    saveloc         = arguments['--out']
    input_string    = arguments['<input_string>']

    _ = docopt_example(input_string, saveloc = saveloc, verbose=verbose)
