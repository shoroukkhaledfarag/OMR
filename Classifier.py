import numpy as np
from functions import *
def classify(output,note,prediction,max_row,min_row,staffspaceheight,stafflineheight,indices):
    #show_images([note])
    head_size=(staffspaceheight)*(staffspaceheight)
    height=max_row-min_row
    
    if (prediction == 11):
        if (np.all(note[(note.shape[0]//2)-2:(note.shape[0]//2)+2,(note.shape[1]//2)-2:(note.shape[1]//2)+2 ]==0)):
              output.append('.')
        else :
            if(max_row-(staffspaceheight/2)>= (5*staffspaceheight)+(5*stafflineheight)+indices[0]):
                output.append('c1/1 ')

            elif(max_row>= (5*staffspaceheight)+(5*stafflineheight)+indices[0]):
                output.append('d1/1 ')

            elif (max_row-(staffspaceheight/2) >= (4*staffspaceheight)+(4*stafflineheight)+indices[0]):
                output.append('e1/1 ')

            elif(max_row>= (4*staffspaceheight)+(4*stafflineheight)+indices[0]):
                output.append('f1/1 ')

            elif(max_row-(staffspaceheight/2)>= (3*staffspaceheight)+(3*stafflineheight)+indices[0]):
                output.append('g1/1 ')

            elif(max_row >= (3*staffspaceheight)+(3*stafflineheight)+indices[0]):
                output.append('a1/1 ')

            elif(max_row-(staffspaceheight/2) >= (2*staffspaceheight)+(2*stafflineheight)+indices[0]):
                output.append('b1/1 ')

            elif(max_row>= (2*staffspaceheight)+(2*stafflineheight)+indices[0]):
                output.append('c2/1')

            elif(max_row-(staffspaceheight/2)>= (1*staffspaceheight)+(1*stafflineheight)+indices[0]):
                output.append('d2/1')

            elif( max_row >= indices[0]+(1*stafflineheight)+(staffspaceheight)):
                output.append('e2/1 ')

            elif( max_row >= indices[0] +(staffspaceheight/2)) :
                output.append('f2/1 ')

            elif (max_row >= indices[0]):
                output.append('g2/1 ')

            elif(max_row >= indices[0]-(staffspaceheight/2)):
                output.append('a2/1 ')

            elif (max_row >= indices[0]-(staffspaceheight)-(staffspaceheight)):
                output.append('b2/1 ')
        
    
    elif(prediction==8):      
        
        heads_num=0
        #print(height//(staffspaceheight))
        for l in range(1,height//(staffspaceheight)):
            segment=note[height-(l)*(staffspaceheight+7):height-(l)*(staffspaceheight+7)+(staffspaceheight+7),:]             
            if(countBlack(segment)>int(head_size)):
                heads_num+=1
        
        #print(heads_num)
        if (heads_num)> 1:
            output.append('{')
            
        if(max_row-(staffspaceheight/2)>= (5*staffspaceheight)+(5*stafflineheight)+indices[0]):
            output.append('c1/4 ')
            if (heads_num)==2:
                output.append('e1/4')
                output.append('}')
            elif(heads_num)==3:
                output.append('e1/4')
                output.append('g1/4')
                output.append('}')

        elif(max_row>= (5*staffspaceheight)+(5*stafflineheight)+indices[0]):
            output.append('d1/4 ')
            if (heads_num)==2:
                output.append('f1/4')
                output.append('}')
            elif(heads_num)==3:
                output.append('f1/4')
                output.append('g1/4')
                output.append('}')

        elif (max_row-(staffspaceheight/2) >= (4*staffspaceheight)+(4*stafflineheight)+indices[0]):
            output.append('e1/4 ')
            if (heads_num)==2:
                output.append('g1/4')
                output.append('}')
            elif(heads_num)==3:
                output.append('g1/4')
                output.append('a1/4')
                output.append('}')

        elif(max_row>= (4*staffspaceheight)+(4*stafflineheight)+indices[0]):
            output.append('f1/4 ')
            if (heads_num)==2:
                output.append('g1/4')
                output.append('}')
            elif(heads_num)==3:
                output.append('g1/4')
                output.append('a1/4')
                output.append('}')

        elif(max_row-(staffspaceheight/2)>= (3*staffspaceheight)+(3*stafflineheight)+indices[0]):
            output.append('g1/4 ')
            if (heads_num)==2:
                output.append('a1/4')
                output.append('}')
            elif(heads_num)==3:
                output.append('a1/4')
                output.append('b1/4')
                output.append('}')


        elif(max_row >= (3*staffspaceheight)+(3*stafflineheight)+indices[0]):
            output.append('a1/4 ')
            if (heads_num)==2:
                output.append('d1/4')
                output.append('}')

        elif(max_row-(staffspaceheight/2) >= (2*staffspaceheight)+(2*stafflineheight)+indices[0]):
            output.append('b1/4 ')

        elif(max_row>= (2*staffspaceheight)+(2*stafflineheight)+indices[0]):
            output.append('c2/4 ')

        elif(max_row-(staffspaceheight/2)>= (1*staffspaceheight)+(1*stafflineheight)+indices[0]):
            output.append('d2/4 ')

        elif( max_row >= indices[0]+(1*stafflineheight)+(staffspaceheight)):
            output.append('e2/4 ')

        elif( max_row >= indices[0] +(staffspaceheight/2)) :
            output.append('f2/4 ')

        elif (max_row >= indices[0]):
            output.append('g2/4 ')

        elif(max_row >= indices[0]-(staffspaceheight/2)):
            output.append('a2/4 ')

        elif (max_row >= indices[0]-(staffspaceheight)-(staffspaceheight)):
            output.append('b2/4 ')
    
    
    
    
            
    elif(prediction==4):
        
        if(max_row-(staffspaceheight/2)>= (5*staffspaceheight)+(5*stafflineheight)+indices[0]):
            output.append('c1/16 ')

        elif(max_row>= (5*staffspaceheight)+(5*stafflineheight)+indices[0]):
            output.append('d1/16 ')

        elif (max_row-(staffspaceheight/2) >= (4*staffspaceheight)+(4*stafflineheight)+indices[0]):
            output.append('e1/16 ')

        elif(max_row>= (4*staffspaceheight)+(4*stafflineheight)+indices[0]):
            output.append('f1/16 ')

        elif(max_row-(staffspaceheight/2)>= (3*staffspaceheight)+(3*stafflineheight)+indices[0]):
            output.append('g1/16 ')

        elif(max_row >= (3*staffspaceheight)+(3*stafflineheight)+indices[0]):
            output.append('a1/16 ')

        elif(max_row-(staffspaceheight/2) >= (2*staffspaceheight)+(2*stafflineheight)+indices[0]):
            output.append('b1/16 ')

        elif(max_row>= (2*staffspaceheight)+(2*stafflineheight)+indices[0]):
            output.append('c2/16 ')

        elif(max_row-(staffspaceheight/2)>= (1*staffspaceheight)+(1*stafflineheight)+indices[0]):
            output.append('d2/16 ')

        elif( max_row >= indices[0]+(1*stafflineheight)+(staffspaceheight)):
            output.append('e2/16 ')

        elif( max_row >= indices[0] +(staffspaceheight/2)) :
            output.append('f2/16 ')

        elif (max_row >= indices[0]):
            output.append('g2/16 ')

        elif(max_row >= indices[0]-(staffspaceheight/2)):
            output.append('a2/16 ')

        elif (max_row >= indices[0]-(staffspaceheight)-(staffspaceheight)):
            output.append('b2/16 ')
            
            
    elif(prediction==9):
                
        if(max_row-(staffspaceheight/2)>= (5*staffspaceheight)+(5*stafflineheight)+indices[0]):
            output.append('c1/8 ')

        elif(max_row>= (5*staffspaceheight)+(5*stafflineheight)+indices[0]):
            output.append('d1/8 ')

        elif (max_row-(staffspaceheight/2) >= (4*staffspaceheight)+(4*stafflineheight)+indices[0]):
            output.append('e1/8 ')

        elif(max_row>= (4*staffspaceheight)+(4*stafflineheight)+indices[0]):
            output.append('f1/8 ')

        elif(max_row-(staffspaceheight/2)>= (3*staffspaceheight)+(3*stafflineheight)+indices[0]):
            output.append('g1/8 ')

        elif(max_row >= (3*staffspaceheight)+(3*stafflineheight)+indices[0]):
            output.append('a1/8 ')

        elif(max_row-(staffspaceheight/2) >= (2*staffspaceheight)+(2*stafflineheight)+indices[0]):
            output.append('b1/8 ')

        elif(max_row>= (2*staffspaceheight)+(2*stafflineheight)+indices[0]):
            output.append('c2/8 ')

        elif(max_row-(staffspaceheight/2)>= (1*staffspaceheight)+(1*stafflineheight)+indices[0]):
            output.append('d2/8 ')

        elif( max_row >= indices[0]+(1*stafflineheight)+(staffspaceheight)):
            output.append('e2/8 ')

        elif( max_row >= indices[0] +(staffspaceheight/2)) :
            output.append('f2/8 ')

        elif (max_row >= indices[0]):
            output.append('g2/8 ')

        elif(max_row >= indices[0]-(staffspaceheight/2)):
            output.append('a2/8 ')

        elif (max_row >= indices[0]-(staffspaceheight)-(staffspaceheight)):
            output.append('b2/8 ')
              
            
        
    elif(prediction==12):
                
        if(max_row-(staffspaceheight/2)>= (5*staffspaceheight)+(5*stafflineheight)+indices[0]):
            output.append('c1/32 ')

        elif(max_row>= (5*staffspaceheight)+(5*stafflineheight)+indices[0]):
            output.append('d1/32 ')

        elif (max_row-(staffspaceheight/2) >= (4*staffspaceheight)+(4*stafflineheight)+indices[0]):
            output.append('e1/32 ')

        elif(max_row>= (4*staffspaceheight)+(4*stafflineheight)+indices[0]):
            output.append('f1/32 ')

        elif(max_row-(staffspaceheight/2)>= (3*staffspaceheight)+(3*stafflineheight)+indices[0]):
            output.append('g1/32 ')

        elif(max_row >= (3*staffspaceheight)+(3*stafflineheight)+indices[0]):
            output.append('a1/32 ')

        elif(max_row-(staffspaceheight/2) >= (2*staffspaceheight)+(2*stafflineheight)+indices[0]):
            output.append('b1/32 ')

        elif(max_row>= (2*staffspaceheight)+(2*stafflineheight)+indices[0]):
            output.append('c2/32 ')

        elif(max_row-(staffspaceheight/2)>= (1*staffspaceheight)+(1*stafflineheight)+indices[0]):
            output.append('d2/32 ')

        elif( max_row >= indices[0]+(1*stafflineheight)+(staffspaceheight)):
            output.append('e2/32 ')

        elif( max_row >= indices[0] +(staffspaceheight/2)) :
            output.append('f2/32 ')

        elif (max_row >= indices[0]):
            output.append('g2/32 ')

        elif(max_row >= indices[0]-(staffspaceheight/2)):
            output.append('a2/32 ')

        elif (max_row >= indices[0]-(staffspaceheight)-(staffspaceheight)):
            output.append('b2/32 ')
        
            
    elif(prediction==6):
        
        if(max_row-(staffspaceheight/2)>= (5*staffspaceheight)+(5*stafflineheight)+indices[0]):
            output.append('c1/2 ')

        elif(max_row>= (5*staffspaceheight)+(5*stafflineheight)+indices[0]):
            output.append('d1/2 ')

        elif (max_row-(staffspaceheight/2) >= (4*staffspaceheight)+(4*stafflineheight)+indices[0]):
            output.append('e1/2 ')

        elif(max_row>= (4*staffspaceheight)+(4*stafflineheight)+indices[0]):
            output.append('f1/2 ')

        elif(max_row-(staffspaceheight/2)>= (3*staffspaceheight)+(3*stafflineheight)+indices[0]):
            output.append('g1/2 ')

        elif(max_row >= (3*staffspaceheight)+(3*stafflineheight)+indices[0]):
            output.append('a1/2 ')

        elif(max_row-(staffspaceheight/2) >= (2*staffspaceheight)+(2*stafflineheight)+indices[0]):
            output.append('b1/2 ')

        elif(max_row>= (2*staffspaceheight)+(2*stafflineheight)+indices[0]):
            output.append('c2/2 ')

        elif(max_row-(staffspaceheight/2)>= (1*staffspaceheight)+(1*stafflineheight)+indices[0]):
            output.append('d2/2 ')

        elif( max_row >= indices[0]+(1*stafflineheight)+(staffspaceheight)):
            output.append('e2/2 ')

        elif( max_row >= indices[0] +(staffspaceheight/2)) :
            output.append('f2/2 ')

        elif (max_row >= indices[0]):
            output.append('g2/2 ')

        elif(max_row >= indices[0]-(staffspaceheight/2)):
            output.append('a2/2 ')

        elif (max_row >= indices[0]-(staffspaceheight)-(staffspaceheight)):
            output.append('b2/2 ')
            
    
    
    elif(prediction==25):
        if(min_row <= indices[0]- (1.5*staffspaceheight)- (2*stafflineheight)  ):
            output.append('b2/4 ') 
            
        elif(min_row<= indices[0]-(staffspaceheight)-(2*stafflineheight)):
            output.append('a2/4 ')
            
        elif(min_row <= indices[0]-(staffspaceheight/2)-(stafflineheight)):
            output.append('g2/4 ')
         
        elif(min_row <= indices[0] -(stafflineheight) ):
            output.append('f2/4 ')
            
        elif( min_row <= indices[0]+ (staffspaceheight/2)):
            output.append('e2/4 ')
            
        elif( min_row <= indices[0]+(staffspaceheight)):
            output.append('d2/4 ')
           
        elif ( min_row <= indices[0]+(stafflineheight)+staffspaceheight+(staffspaceheight/2)):
            output.append('c2/4 ') 
            
        elif ( min_row <= (2*staffspaceheight)+(1*stafflineheight)+indices[0]):
            output.append('b1/4 ') 
        
        elif ( min_row <=( 2.5*staffspaceheight)+(2*stafflineheight)+indices[0]):
            output.append('a1/4 ') 
        
        elif ( min_row <=( 3*staffspaceheight)+(2*stafflineheight)+indices[0]):
            output.append('g1/4 ') 
            
        elif ( min_row <=( 3.5*staffspaceheight)+(3*stafflineheight)+indices[0]):
            output.append('f1/4 ')
        
        elif ( min_row <=( 4*staffspaceheight)+(3*stafflineheight)+indices[0]):
            output.append('e1/4 ') 
            
        elif ( min_row <=( 4.5*staffspaceheight)+(4*stafflineheight)+indices[0]):
            output.append('d1/4 ') 
        
        elif ( min_row <=( 5*staffspaceheight)+(4*stafflineheight)+indices[0]):
            output.append('c1/4 ')      
            
 
            
    elif(prediction==26):
        
        if(min_row <= indices[0]- (1.5*staffspaceheight)- (2*stafflineheight)  ):
            output.append('b2/2 ') 
            
        elif(min_row<= indices[0]-(staffspaceheight)-(2*stafflineheight)):
            output.append('a2/2 ')
            
        elif(min_row <= indices[0]-(staffspaceheight/2)-(stafflineheight)):
            output.append('g2/2 ')
         
        elif(min_row <= indices[0] -(stafflineheight) ):
            output.append('f2/2 ')
            
        elif( min_row <= indices[0]+ (staffspaceheight/2)):
            output.append('e2/2')
            
        elif( min_row <= indices[0]+(staffspaceheight)):
            output.append('d2/2')
           
        elif ( min_row <= indices[0]+(stafflineheight)+staffspaceheight+(staffspaceheight/2)):
            output.append('c2/2 ') 
            
        elif ( min_row <= (2*staffspaceheight)+(1*stafflineheight)+indices[0]):
            output.append('b1/2 ') 
        
        elif ( min_row <=( 2.5*staffspaceheight)+(2*stafflineheight)+indices[0]):
            output.append('a1/2 ') 
        
        elif ( min_row <=( 3*staffspaceheight)+(2*stafflineheight)+indices[0]):
            output.append('g1/2 ') 
            
        elif ( min_row <=( 3.5*staffspaceheight)+(3*stafflineheight)+indices[0]):
            output.append('f1/2 ')
        
        elif ( min_row <=( 4*staffspaceheight)+(3*stafflineheight)+indices[0]):
            output.append('e1/2 ') 
            
        elif ( min_row <=( 4.5*staffspaceheight)+(4*stafflineheight)+indices[0]):
            output.append('d1/2') 
        
        elif ( min_row <=( 5*staffspaceheight)+(4*stafflineheight)+indices[0]):
            output.append('c1/2 ')        
                       
            
            
            
    elif(prediction==17):
        note=np.asarray(note)
        img_part1 = np.asarray(note[:,0:note.shape[1]//2])
        segmented_part1,min_r_seg1,max_r_seg1 =segmentation(img_part1)
        knn_prediction_persegment = 9
        row_up = int(min_row)+int(max_r_seg1[0])
        row_low = int(min_row)+int(min_r_seg1[0])
        
        string=classify(output,note,knn_prediction_persegment,row_up,row_low,staffspaceheight,stafflineheight ,indices)
#         output.append(string)
        
        img_part2 = np.asarray(note[:,note.shape[1]//2:note.shape[1]])
        segmented_part2,min_r_seg2,max_r_seg2 =segmentation(img_part2)

        knn_prediction_persegment = 9
        row_up=int(min_row)+int(max_r_seg2[0])
        row_low = int(min_row)+int(min_r_seg2[0])
        
        string2 = classify(output,note,knn_prediction_persegment,row_up,row_low,staffspaceheight,stafflineheight,indices)
#         output.append(string2)
        
        
        
    elif(prediction==22):
        
        for j in range(4):   
            img = note[:,j*note.shape[1]//4 : (j+1)*note.shape[1]//4 ]
            segmented,min_r_seg,max_r_seg =segmentation(img)
            knn_prediction_persegment = 4
            row_up = int(min_row)+int(max_r_seg[0])
            row_low = int(min_row)+int(min_r_seg[0])
        
            string=classify(output,note,knn_prediction_persegment,row_up,row_low,staffspaceheight,stafflineheight,indices )
#             output.append(string)

    elif(prediction==19):
        
        for j in range(3):   
            img = note[:,j*note.shape[1]//3 : (j+1)*note.shape[1]//3]
            segmented,min_r_seg,max_r_seg =segmentation(img)
            knn_prediction_persegment = 4
            row_up = int(min_row)+int(max_r_seg[0])
            row_low = int(min_row)+int(min_r_seg[0])
        
            string=classify(output,note,knn_prediction_persegment,row_up,row_low,staffspaceheight,stafflineheight,indices )
#             output.append(string)   
    
    
    elif(prediction==7):
        output.append("\meter<'4/4'>")
        
    elif(prediction==10):
        if (getVolume(note[:note.shape[0]//2,:note.shape[1]//2]))<=0.55:
            output.append('#')
            
    elif(prediction==2):
        output.append('&&')
            
    elif(prediction==3):
        output.append('##')
            
    elif(prediction==5):
         output.append('&')

        
    else:
        output.append(str(" "))
    return output
