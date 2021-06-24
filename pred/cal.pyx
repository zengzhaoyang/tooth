import numpy as np
cimport numpy as np
import queue

DTYPE = np.int
ctypedef np.int_t DTYPE_t

def cal(np.ndarray segs):

    cdef int x = segs.shape[0]
    cdef int y = segs.shape[1]
    cdef int z = segs.shape[2]


    tmp = np.zeros_like(segs)
    cdef int nowcnt = 1

    q = queue.Queue()
    qlen = 0 
    dx = [1,-1,0,0,0,0]
    dy = [0,0,1,-1,0,0]
    dz = [0,0,0,0,1,-1]
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if segs[i,j,k] != 0 and tmp[i, j, k] == 0:
#                    print(i,j,k)
                    q.put((i,j,k))
                    tmp[i,j,k] = nowcnt
                    qlen += 1
                    while qlen > 0:
                        ni, nj, nk = q.get()
                        qlen -= 1
                        for ii in range(6):
                            newi = ni + dx[ii]
                            newj = nj + dy[ii]
                            newk = nk + dz[ii]
                            if newi >= 0 and newi < x and newj >=0 and newj < y and newk >= 0 and newk < z and segs[newi, newj, newk] != 0 and tmp[newi, newj, newk] == 0:
                                tmp[newi, newj, newk] = nowcnt
                                q.put((newi, newj, newk))
                                qlen += 1
                    nowcnt += 1
    

    return tmp, nowcnt
