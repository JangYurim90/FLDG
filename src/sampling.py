import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """

    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {},[i for i in range(len(dataset))]
    for i in range(num_users): #모든 인덱스에서 item의 수만큼의 개수를 추출 (set으로 묶어주면서 중복제거)
        dict_users[i] = set(np.random.choice(all_idxs,num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i]) #유저의 추출데이터를 중복되지 않도록 set에서 빼준다.
    return dict_users

def mnist_noniid(dataset, num_users):
    num_shards, num_imgs = 200,300
    idx_shard = [i for i in range(num_shards)] # list에 shrads의 개수만큼 index 생성
    dict_users = {i : np.array([]) for i in range(num_users)} # 유저의 수 만큼 딕셔너리 생성
    idxs = np.arange(num_shards*num_imgs) # shard이 개수와 이미지의 개수를 곱한 수 만큼의 list 생성 (200*300=> [0,1,,,,,59999]
    labels = dataset.train_labels.numpy() # dataset.train_labels를 numpy 배열로 반환한다.

    #labels 정렬(sort)
    idxs_labels = np.vstack((idxs,labels)) #idx와 labels의 결합 ([[idxs],[labels]])
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()] #idxs_labels에서 index는 그대로 두고 labels만 오름차순으로 정렬
    idxs = idxs_labels[0, :] #idxs는 따로 정렬된 새로 운 idx로 저장

    #각 client당 2개의 shards로 분배
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace = False))
        idx_shard = list(set(idx_shard)-rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i],idxs[rand*num_imgs:(rand+1)*num_imgs]),axis = 0)
            # numpy 합침 // {각 유저 딕셔너리 index i : 뽑아진 각 shard인덱스에 맞게 imgs 개수를 곱해서 해당 index를 함께 저장}
    return dict_users  # 딕셔너리 유저 반납
