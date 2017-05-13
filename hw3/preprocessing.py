import csv

tag_file="tags_clean.csv"
tag_threshold=1000 # unwanted tag with number less than threshold
def tag_preproc():
    def num_in_str(string):
        ix_colon=string.index(':')
        word=string[:ix_colon]
        word=word.replace(' ','_')
        num =int(string[ix_colon+1:])
        return  word , num
    with open(tag_file,'rb') as tag_f:
        tags=csv.reader(tag_f)
        captions=[]
        for i,tag in enumerate(tags):
            tag_split=[ num_in_str(string)[0] for string in tag[1].split('\t') if string!='' and num_in_str(string)[1]>tag_threshold ]
            caption=' '.join(tag_split)
            print caption
            print i
            captions+=caption

    return captions
if __name__ == '__main__':
    cap=tag_preproc()
    print cap[1]

