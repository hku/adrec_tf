import numpy as np
from scipy.sparse import csr_matrix, hstack

def search_range(vmin, vmax, ol):

	def search_low_boundary(v, ol):
		low = 0
		high = len(ol)
		assert v >= ol[low], "v is out of low boundary" 

		while high > low:
			mid = (low + high)//2
			if ol[mid] > v:
				if high == mid:
					break
				high = mid
			elif ol[mid] < v:
				if low == mid:
					break
				low = mid
			else:
				return mid, ol[mid]
		return low, ol[low]


	def search_high_boundary(v, ol):
		low = 0
		high = len(ol)
		assert v <= ol[high - 1], "v is out of high boundary" 

		while high > low:
			mid = (low + high)//2
			if ol[mid] > v:
				if high == mid:
					break
				high = mid
			elif ol[mid] < v:
				if low == mid:
					break
				low = mid
			else:
				return mid, ol[mid]
		return high, ol[high]


	if vmin > vmax:
		vmin, vmax = vmax, vmin
	idx1, _ = search_high_boundary(vmin, ol)
	idx2, _ = search_low_boundary(vmax, ol)

	# assert idx2 >= idx1, "idx2 should larger than idx1"
	# 
	return idx1, idx2, None if idx2>=idx1 else 1


def read_raw_data(fname):
	with open(fname) as f:
		raw_features, raw_target = zip(*[ (np.fromstring(x, sep=" ", dtype=np.int), int(y)) for (x, y) in [line.strip().split("|") for line in f] ])
	return np.array(raw_features), np.array(raw_target)


def get_csr_matrix(xx_all, col_num):
	row_num = len(xx_all)
	indices = [[i, j] for i, xx in enumerate(xx_all) if isinstance(xx, np.ndarray) for j in xx]


	row_indices, col_indices = zip(*indices) if len(indices)!=0 else ([], [])
	values = np.ones(shape=(len(row_indices),), dtype=np.float32)
	shape = (row_num, col_num)

	return csr_matrix((values, (row_indices, col_indices)), shape=shape)


def get_selected_input(raw_features, cols, cfg):
	ufeatures = cfg["ufeatures"]
	adfeatures = cfg["adfeatures"]
	cards, names = zip(*map(lambda i: (i["cardinality"], i["name"]), np.concatenate([ufeatures, adfeatures])))
	cards_cum = np.cumsum(np.insert(cards, 0, 0, axis=0))
		
	col_xx_all = []

	col_cardinalities = []

	if isinstance(cols, str):
		if cols == "ad_all":
			cols = map(lambda i: i["name"], adfeatures)
		elif cols == "u_all":
			cols = map(lambda i: i["name"], ufeatures)

	for col in cols:
		col_idx = names.index(col)
		vmin = cards_cum[col_idx]
		vmax = cards_cum[col_idx + 1] - 1
		res = []
		for xx in raw_features:
			idx1, idx2, err = search_range(vmin, vmax, xx)
			if not err:
				res.append(xx[idx1:idx2 + 1] - vmin)
			else:
				res.append(-1)

		col_xx_all.append(res)
		col_cardinalities.append(cards[col_idx])
	matrixes = [get_csr_matrix(xx_all, col_num) for xx_all, col_num in zip(col_xx_all, col_cardinalities)]

	xx_sparse_all =  hstack(matrixes)

	return xx_sparse_all.tocsr()