# for template based model test
python retro_plan.py --seed 42 --use_value_fn \
--expansion_topk 8 --iterations 101 \
--one_step_type template_based --viz --gpu 0 \
--test_routes uspto190


# for template free model test

# with CSS 9 and RD and DICT
python retro_plan.py --seed 42 --use_value_fn --viz --gpu 1 \
--expansion_topk 16 --iterations 201 \
--one_step_type template_free --CSS --RD_list "[(7,2),(3,0)]"  --DICT  \
--test_routes 8XIK_NCI


# --test_routes uspto190 | pth_hard | 8XIK | olorofim |8XIK_olorofim
# --RD_list "[(7,2),(3,0)]" | "[(7,2)]" 
# without CSS, RD and DICT
python retro_plan.py --seed 42 --use_value_fn --viz --gpu 1 \
--expansion_topk 8 --iterations 101 \
--one_step_type template_free  \
--test_routes uspto190


python retro_plan.py --seed 42 --use_value_fn --viz --gpu 0 \
--expansion_topk 8 --iterations 101 \
--one_step_type template_free  \
--test_routes pth_hard


bash scripts/run_template_free_batch_impact.sh  --gpu 0 --seed 42 --iterations 10 --expansion-topk 8  --max-targets 10 --test-routes uspto190 --parallel-num 4 --repeats 1  --rd-list "[(7,2),(3,0)]"