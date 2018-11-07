
cid = [int(n) for n in '935517']

def CID(t):
    return cid[t-1]

def s(t):
    return (CID(t) + 1) % 3

def r(t):
    return CID(t) % 2

def trace(cid):
    res = []
    for t in range(1, len(cid)+1):
        temp = s(t), r(t)
        res.append(temp)
    return res

if __name__ == '__main__':
    # q1.1
    res = trace(cid) 
    print(res)
    
