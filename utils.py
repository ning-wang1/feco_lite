

def cal_score(model, normal_vec, out):
    """
    Generate and save scores
    """

    sim_1 = torch.mm(out, normal_vec.t())


    return sim_1_list.numpy()