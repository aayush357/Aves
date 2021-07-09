def list_birds():
    f=open(r'text_files/birdNames.txt')
    x=f.read()
    l1=x.split()
    l=len(l1)
    bird_names=[]
    for i in range(1,l,2):
        bird_names.append(l1[i])

    return bird_names
