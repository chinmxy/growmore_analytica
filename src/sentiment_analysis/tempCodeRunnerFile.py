    if v!='[]':            
            wordcloud = WordCloud().generate(str(dictionary[k]))                
            fig, axes= plt.subplots(figsize=(20,12),clear=True)                     
            plt.imshow(wordcloud, interpolation='bilinear')            
            plt.show()                 
        else:            
            print(str(k[0])+"_"+str(k[1][5:10])+"_"+str(k[1][11:13])              
            +"_"+str(k[1][14:16]) +"_"+str(k[1][17:19])+"_"+str(k[2]))             
            print("Wordcloud Not applicable")