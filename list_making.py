def list_birds():
    f=open(r'text_files/birdNames.txt')
    x=f.read()
    l1=x.split()
    l=len(l1)

    list_to_use=[]
    for i in range(1,l,2):
        list_to_use.append(l1[i])

    return(list_to_use)
