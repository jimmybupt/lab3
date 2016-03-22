from optparse import OptionParser

def test():
	print "test"

def initialize_parser(parser):
	parser.add_option("-f", "--file", dest="in_file",
		 help="the input vector",metavar="FILE",default="vector1.txt")
	parser.add_option("-o", "--output", dest="out_file",
		  help="the output matrix file", metavar="FILE")
	parser.add_option("-m", "--metric", dest="metric",
		  metavar="[Euclidean|Manhattan]",
		 default="euclidean", help="Type of similarity")
	parser.add_option("-a", "--algorithm", dest="algorithm",
		  metavar="[DBSCAN|Hierarchical]",
		  default="DBSCAN")
	parser.add_option("-e", "--eps", dest="epsilon", metavar="<Epsilon>",
		  type="float", default=0.5)
	parser.add_option("-M", "--min-sample", dest="min_sample", metavar="<Min samples>",
		  type="int", default=5) 
	parser.add_option("-t", "--test", dest="small_data",
		  action="store_true", default=False)
	parser.add_option("-k", dest="cluster", metavar="<Cluster>", type="int",
		  default=2)
	parser.add_option("-l", dest="linkage", metavar="[Ward|Average|Complete]",
		  default="ward")
	
	return parser
