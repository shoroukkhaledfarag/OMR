from functions import *
from Classifier import *
import fileinput



def operate(folder_in,folder_out):
    files = glob.glob (folder_in+"/*")
    for file in files:

        print("-----------------------",file,"------------------------------")
        img=io.imread(file)

        gray_scale_img = (rgb2gray(img)*255).astype(np.uint8)


        gaussian = filters.gaussian(gray_scale_img,sigma =0.6 )

        gaussianbinarized = binarize(gaussian)

        deskewImage_gauss,is_rotated_guass = deskew(gaussianbinarized)

        gaussianbinarized = binarize(deskewImage_gauss)


        th1=feng_threshold(gray_scale_img)
        img_1=to_binary(gray_scale_img,th1)


        adaptivebinarized = binarize(gray_scale_img)

        img_1=adaptivebinarized*img_1

        deskewImage_adap,is_rotated_adap = deskew(img_1.astype('float'))


        deskewImage_adap=binarize(deskewImage_adap)

        difference_row =100
        difference_col = 100
        deskewImage_adap= deskewImage_adap[difference_row:(deskewImage_adap.shape[0] - difference_row),difference_col : (deskewImage_adap.shape[1] - difference_col)]


        if(is_rotated_adap==True):
            sub_img,staffspaceheight,stafflineheight = staff_removal_2(deskewImage_adap)
            stacks_segmented , stacks_min_r , stacks_max_r , stacks_indices = segmentation_2(sub_img,stafflineheight)

        else:
            sub_img,staffspaceheight,stafflineheight = staff_removal_1(gaussianbinarized)
            stacks_segmented , stacks_min_r , stacks_max_r , stacks_indices = segmentation_1(sub_img,stafflineheight)
        s=0
        output_string=[]
        flag=0
        name = file.split("/")[-1].split(".")[0]
        f = open( folder_out+"/"+name+".txt", "w+")

        for stack in stacks_segmented:
            z=0
            output_string=[]
            for note in stack:
                    feature_vector=get_feature_vector(note)

                    training_data = read_data('dataset.csv')
                    X = training_data[:,[4,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]]

                    X_Test =np.asarray(feature_vector)
                    Y = training_data[:,0]

                    k = 3 
                    knn_prediction = KNN(X_Test,X,Y,k)
                    print('class = ',knn_prediction)

                    max_r=stacks_max_r[s][z]
                    min_r = stacks_min_r[s][z]
                    indices = stacks_indices[s]
                    string=classify(output_string,note,knn_prediction,max_r,min_r,staffspaceheight,stafflineheight,indices)


                    z+=1

            print(output_string)



            if len(stacks_segmented) >1 and flag == 0 :
                f.write("{")
                f.write("\n")
                flag=1

            f.write("[") 
            for str1 in output_string:
                    f.write(str1)
            f.write("]") 
            f.write("\n")

            s+=1


        if len(stacks_segmented) >1 :
                f.write("}")
                f.close()
        else:
            f.close()



        #read input file
        fin = open(folder_out+"/"+name+".txt", "rt")
        #read file contents to string
        data = fin.read()
        #replace all occurrences of the required string
        data = data.replace('##]', ']')

        data = data.replace('##e', 'e##')
        data = data.replace('##c', 'c##')
        data = data.replace('##d', 'd##')
        data = data.replace('##e', 'e##')
        data = data.replace('##f', 'f##')
        data = data.replace('##b', 'b##')
        data = data.replace('##a', 'a##')
        data = data.replace('##g', 'g##')




        data = data.replace('#e', 'e#')
        data = data.replace('#c', 'c#')
        data = data.replace('#d', 'd#')
        data = data.replace('#e', 'e#')
        data = data.replace('#f', 'f#')
        data = data.replace('#b', 'b#')
        data = data.replace('#a', 'a#')
        data = data.replace('#g', 'g#')

        data = data.replace('&&e', 'e&&')
        data = data.replace('&&c', 'c&&')
        data = data.replace('&&d', 'd&&')
        data = data.replace('&&e', 'e&&')
        data = data.replace('&&f', 'f&&')
        data = data.replace('&&b', 'b&&')
        data = data.replace('&&a', 'a&&')
        data = data.replace('&&g', 'g&&')

        data = data.replace('&e', 'e&')
        data = data.replace('&c', 'c&')
        data = data.replace('&d', 'd&')
        data = data.replace('&e', 'e&')
        data = data.replace('&f', 'f&')
        data = data.replace('&b', 'b&')
        data = data.replace('&a', 'a&')
        data = data.replace('&g', 'g&')


        #close the input file
        fin.close()
        #open the input file in write mode
        fin = open(folder_out+"/"+name+".txt", "wt")
        #overrite the input file with the resulting data
        fin.write(data)
        #close the file
        fin.close()
    return True
