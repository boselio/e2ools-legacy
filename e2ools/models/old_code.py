def update_sticks_new_jump_update(tp_initial, interactions, alpha, theta):

    num_recs = len(set([r for t, recs in interactions for r in recs]))
    recs_initial, change_times = zip(*[(r, t) for (r, v) in tp_initial.arrival_times_dict.items() 
                                        for t in v[1:]])
    change_times = list(change_times)
    recs_initial = list(recs_initial)


    if len(tp_initial.arrival_times_dict[-1]) > 0:
        change_times.append(tp_initial.arrival_times_dict[-1][0])
        recs_initial.append(-1)

    change_times = np.array(change_times)
    sorted_inds = change_times.argsort()
    change_times = change_times[sorted_inds]
    recs_initial = np.array(recs_initial)[sorted_inds]

    rec_choice = np.zeros_like(change_times)
    stick_choice = np.zeros_like(change_times)
    interaction_times = np.array([interaction[0] for interaction in interactions])
    max_time = interactions[-1][0]
    #created_set = set()

    permuted_inds = np.random.permutation(len(change_times))
    
    # calculate all degrees between change times for all receivers
    degree_mat = np.zeros((num_recs, len(change_times) + 1))
    beta_mat = np.zeros((num_recs, len(change_times) + 1))

    for i, (begin_time, end_time) in enumerate(zip(np.concatenate([[0], change_times]), np.concatenate([change_times, [interaction_times[-1] + 1]]))):

        begin_ind = bisect_left(interaction_times, begin_time)
        end_ind = bisect_right(interaction_times, end_time)
        if begin_ind == end_ind:
            continue
            
        recs, degrees = np.unique([r for interaction in interactions[begin_ind:end_ind] for r in interaction[1]],
                              return_counts=True)

        for r in recs:
            if begin_time >= tp_initial.created_times[r] and end_time <= tp_initial.created_times[r]:
                degrees[recs == r] -= 1

        try:
            degree_mat[recs, i] = degrees
        except IndexError:
            import pdb
            pdb.set_trace()

        for r in range(num_recs):
            beta_mat[r, i] = tp_initial.get_stick(r, end_time)

    s_mat = np.vstack([np.flipud(np.cumsum(np.flipud(degree_mat), axis=0))[1:, :], 
           np.zeros((1, len(change_times)+1))])

    for ind in permuted_inds:
    #Need to calculate, the likelihood of each stick if that receiver
    #was not chosen.

        ct = change_times[ind]
        try:
            end_time = change_times[ind+1]
        except: end_time = interaction_times[-1] + 1
        if ind != 0:
            begin_time = change_times[ind - 1]
        else:
            begin_time = 0
        for r in range(num_recs):
            beta_mat[r, ind] = tp_initial.get_stick(r, ct, before=True)
            beta_mat[r, ind+1] = tp_initial.get_stick(r, ct)
     
        num_created_recs = len(tp_initial.created_times[tp_initial.created_times < ct])
        probs = np.array([tp_initial.get_stick(r, ct) for r in range(num_created_recs)] + [1])
        probs[1:] = probs[1:] * np.cumprod(1 - probs[:-1])

        log_probs = np.log(probs)
        #Calculate likelihood of each jump
        #First step, add integrated new beta
        log_probs[:-1] += betaln(1 - alpha + degree_mat[:num_created_recs, ind+1], 
                            theta + np.arange(1, num_created_recs+1) * alpha + s_mat[:num_created_recs, ind+1])
        
        #I think this next line is wrong.
        #log_probs[-1] += betaln(1 - alpha, 
        #                    theta + num_created_recs+1 * alpha)

        #Now, need to add all other likelihood components, i.e. all degrees for
        #which the receiver did not jump.
        before_likelihood_components = degree_mat[:num_created_recs, ind] * np.log(beta_mat[:num_created_recs, ind])
        before_likelihood_components += s_mat[:num_created_recs, ind] * np.log(1 - beta_mat[:num_created_recs, ind])

        after_likelihood_components = degree_mat[:num_created_recs, ind+1] * np.log(beta_mat[:num_created_recs, ind+1])
        after_likelihood_components += s_mat[:num_created_recs, ind+1] * np.log(1 - beta_mat[:num_created_recs, ind+1])

        log_probs += np.sum(before_likelihood_components)
        log_probs += np.sum(after_likelihood_components)
        log_probs[:-1] -= after_likelihood_components
        #log_probs[-1] += np.sum(likelihood_components)

        probs = np.exp(log_probs - logsumexp(log_probs))

        try:
            new_choice = np.random.choice(num_created_recs+1, p=probs)
        except ValueError:
            import pdb
            pdb.set_trace()
        rec_choice[ind] = new_choice
        if new_choice == recs_initial[ind]:
            if new_choice == num_created_recs:
                #Do nothing, it stayed in the tail
                continue
            else:
                #Draw the beta
                end_time = tp_initial.get_next_switch(new_choice, ct)
                if end_time == -1:
                    end_time = max_time
                begin_ind = bisect_left(interaction_times, ct)
                end_ind = bisect_right(interaction_times, end_time)

                new_stick = draw_beta(interactions[begin_ind:end_ind], tp_initial, ct, alpha, theta, new_choice)

                change_index = tp_initial.arrival_times_dict[new_choice].index(ct)
                tp_initial.stick_dict[new_choice][change_index] = new_stick

        else:
            r_delete = int(recs_initial[ind])
            tp_initial.delete_change(r_delete, ct)
        
            if r_delete != -1:
                # redraw the beta that we had deleted.
                begin_time, change_ind = tp_initial.get_last_switch(r_delete, ct, return_index=True)
                end_time = tp_initial.get_next_switch(r_delete, ct)
                if end_time == -1:
                    end_time = max_time

                begin_ind = bisect_left(interaction_times, begin_time)
                end_ind = bisect_right(interaction_times, end_time)

                new_stick = draw_beta(interactions[begin_ind:end_ind], tp_initial, begin_time, alpha, theta, r_delete)

                tp_initial.stick_dict[r_delete][change_ind] = new_stick

            if new_choice == num_created_recs:
                rec_choice[ind] = -1
                stick_choice[ind] = -1
                tp_initial.insert_change(-1, ct, -1.0)

            else:
                # Draw the beta backward
                begin_time, change_ind = tp_initial.get_last_switch(new_choice, ct, return_index=True)
                begin_ind = bisect_left(interaction_times, begin_time)
                end_ind = bisect_right(interaction_times, ct)
                
                new_stick = draw_beta(interactions[begin_ind:end_ind], tp_initial, begin_time, alpha, theta, new_choice)

                tp_initial.stick_dict[new_choice][change_ind] = new_stick

                #Draw the beta forward
                end_time = tp_initial.get_next_switch(new_choice, ct)
                if end_time == -1:
                    end_time = max_time
                begin_ind = bisect_left(interaction_times, ct)
                end_ind = bisect_right(interaction_times, end_time)

                new_stick = draw_beta(interactions[begin_ind:end_ind], tp_initial, ct, alpha, theta, new_choice)

                tp_initial.insert_change(new_choice, ct, new_stick)

    # Reupdate all the initial sticks, in case they did not get updated.
    for r in range(num_recs):
            #draw beta
        end_time = tp_initial.get_next_switch(r, tp_initial.created_times[r])
        if end_time == -1:
            end_time = max_time

        begin_ind = bisect_left(interaction_times, tp_initial.created_times[r])
        end_ind = bisect_right(interaction_times, end_time)

        new_stick = draw_beta(interactions[begin_ind:end_ind], tp_initial, tp_initial.created_times[r], alpha, theta, r)

        tp_initial.stick_dict[r][0] = new_stick

    return tp_initial, rec_choice, stick_choice


def update_sticks_v2(tp_initial, interactions, alpha, theta):

    num_recs = len(set([r for t, recs in interactions for r in recs]))
    recs_initial, change_times = zip(*[(r, t) for (r, v) in tp_initial.arrival_times_dict.items() 
                                        for t in v[1:]])
    change_times = list(change_times)
    recs_initial = list(recs_initial)


    if len(tp_initial.arrival_times_dict[-1]) > 0:
        change_times.append(tp_initial.arrival_times_dict[-1][0])
        recs_initial.append(-1)

    change_times = np.array(change_times)
    sorted_inds = change_times.argsort()
    change_times = change_times[sorted_inds]
    recs_initial = np.array(recs_initial)[sorted_inds]

    rec_choice = np.zeros_like(change_times)
    stick_choice = np.zeros_like(change_times)
    interaction_times = np.array([interaction[0] for interaction in interactions])
    max_time = interactions[-1][0]
    #created_set = set()

    permuted_inds = np.random.permutation(len(change_times))

    for ind in permuted_inds:

        ct = change_times[ind]

        num_created_recs = len(tp_initial.created_times[tp_initial.created_times < ct])
        probs = np.array([tp_initial.get_stick(r, ct) for r in range(num_created_recs)] + [1])
        probs[1:] = probs[1:] * np.cumprod(1 - probs[:-1])

        new_choice = np.random.choice(num_created_recs+1, p=probs)

        if new_choice == recs_initial[ind]:
            if new_choice == num_created_recs:
                #Do nothing
                continue
            #Draw the beta
            end_time = tp_initial.get_next_switch(new_choice, ct)
            if end_time == -1:
                end_time = max_time
            begin_ind = bisect_left(interaction_times, ct)
            end_ind = bisect_right(interaction_times, end_time)

            new_stick = draw_beta(interactions[begin_ind:end_ind], tp_initial, ct, alpha, theta, new_choice)

            change_index = tp_initial.arrival_times_dict[new_choice].index(ct)
            tp_initial.stick_dict[new_choice][change_index] = new_stick

        else:
            r_delete = int(recs_initial[ind])
            tp_initial.delete_change(r_delete, ct)
        
            if r_delete != -1:
                # redraw the beta that we had deleted.
                begin_time, change_ind = tp_initial.get_last_switch(r_delete, ct, return_index=True)
                end_time = tp_initial.get_next_switch(r_delete, ct)
                if end_time == -1:
                    end_time = max_time

                begin_ind = bisect_left(interaction_times, begin_time)
                end_ind = bisect_right(interaction_times, end_time)

                new_stick = draw_beta(interactions[begin_ind:end_ind], tp_initial, begin_time, alpha, theta, r_delete)

                tp_initial.stick_dict[r_delete][change_ind] = new_stick

            if new_choice == num_created_recs:
                rec_choice[ind] = -1
                stick_choice[ind] = -1
                tp_initial.insert_change(-1, ct, -1.0)

            else:
                # Draw the beta backward
                begin_time, change_ind = tp_initial.get_last_switch(new_choice, ct, return_index=True)
                begin_ind = bisect_left(interaction_times, begin_time)
                end_ind = bisect_right(interaction_times, ct)
                
                new_stick = draw_beta(interactions[begin_ind:end_ind], tp_initial, begin_time, alpha, theta, new_choice)

                tp_initial.stick_dict[new_choice][change_ind] = new_stick

                #Draw the beta forward
                end_time = tp_initial.get_next_switch(new_choice, ct)
                if end_time == -1:
                    end_time = max_time
                begin_ind = bisect_left(interaction_times, ct)
                end_ind = bisect_right(interaction_times, end_time)

                new_stick = draw_beta(interactions[begin_ind:end_ind], tp_initial, ct, alpha, theta, new_choice)

                tp_initial.insert_change(new_choice, ct, new_stick)

    # Reupdate all the initial sticks, in case they did not get updated.
    for r in range(num_recs):
            #draw beta
        end_time = tp_initial.get_next_switch(r, tp_initial.created_times[r])
        if end_time == -1:
            end_time = max_time

        begin_ind = bisect_left(interaction_times, tp_initial.created_times[r])
        end_ind = bisect_right(interaction_times, end_time)

        new_stick = draw_beta(interactions[begin_ind:end_ind], tp_initial, tp_initial.created_times[r], alpha, theta, r)

        tp_initial.stick_dict[r][0] = new_stick

    return tp_initial, rec_choice, stick_choice
