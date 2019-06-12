for method in "SRGSD"
do
    #for dataset in "Set5" "Set14" "B100" "Manga109"
    for dataset in "Set5"
    do
        for scale in 2
        do
            for val in "BI_0" "BI_2" "BI_50"  # D1 , D2 , D3
            do
                name="_BIX"
                python main.py --data_test MyImage --scale $scale --model $method --pre_train ../model/$method/${method}${name}${scale}.pt --test_only --save_results --chop --save $method --testpath ../LR --testset $dataset --bi_blur_path $val
            done
        done
    done
done
