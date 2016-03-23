from optparse import OptionParser

def initialize_parser(parser):
	parser.add_option("-f", "--file", dest="in_file",
		 help="the input vector",metavar="FILE",default="vector1.txt")
	parser.add_option("-o", "--output", dest="out_file",
		  help="the output matrix file", metavar="FILE")
	parser.add_option("-m", "--model", dest="model",
		  metavar="[KNN|DecisionTree|Bayesian]",
		 default="DecisionTree", help="Type of classifier")
	return parser
