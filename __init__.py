
from .NetTS import *



if __name__ == '__main__':
	from NetTS import *

	from ut_measure import *
	from ut_ntsfiles import *
	from ut_getsetedges import *

	# run unit tests
	ut_measure()
	ut_ntsfiles()
	ut_getsetedges()

