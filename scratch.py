def chopper_sliding(word_list,my_senti_dict,min_size=10000,num_points=100,stop_val=0.0):
    """Take long piece of text and generate the sentiment time series.
    We will now slide the window along, rather than make uniform pieces.
    """
    step = int(floor((len(word_list)-min_size)/(num_points-1)))
    centers = [i*step+(min_size)/2 for i in range(num_points)]
    all_fvecs = lil_matrix((num_points,len(my_senti_dict.data)),dtype="i")
    for i in range(num_points-1):
        window_dict = dict()
        # print("using words {} through {}".format(i*step,min_size+i*step))
        for word in word_list[(i*step):(min_size+i*step)]:
            if word in window_dict:
                window_dict[word] += 1
            else:
                window_dict[word] = 1
        text_fvec = my_senti_dict.wordVecify(window_dict)
        all_fvecs[i,:] = text_fvec
    i = num_points-1
    window_dict = dict()
    # print("using words {} through {}".format(i*step,len(all_words)))
    for word in word_list[(i*step):]:
        if word in window_dict:
            window_dict[word] += 1
        else: 
           window_dict[word] = 1
    text_fvec = my_senti_dict.wordVecify(window_dict)
    all_fvecs[i,:] = text_fvec
    all_fvecs = all_fvecs.tocsr()
    
    timeseries = [0 for i in range(num_points)]
    
    for i in range(num_points):
        text_fvec = all_fvecs[i,:].toarray().squeeze()
        stoppedVec = stopper(text_fvec,my_senti_dict.scorelist,my_senti_dict.wordlist,stopVal=stop_val)
        timeseries[i] = dot(my_senti_dict.scorelist,stoppedVec)/sum(stoppedVec)

    return all_fvecs,timeseries
