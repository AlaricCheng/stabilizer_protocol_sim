conda activate galois

python ./fail_attack.py -n 750 -m 1200 -g 10 -m1 300 -d 143 -d1 447 --C2-type bch --dump
python ./distance.py --uuid 692ca9cc-f7de-4e4b-a00c-f049aefebf18 -g 10 -d 143

python ./fail_attack.py -n 750 -m 1200 -g 10 -m1 300 -d 143 -d1 447 --C2-type doubly-even --dump
python ./distance.py --uuid 9a785303-0a01-4f80-8496-37e53d0eeb5b -g 10 -d 143


python ./distance.py --compare -g 9 -d 131 --m1 277 --m2 1023 --r2 638

python ./distance.py --compare -g 10 -d 143 --m1 300 --m2 900 --r2 597

python ./fail_attack.py -n 728 -m 1200 -g 10 -m1 300 -d 98 -d1 0 --C2-type orthogonal --dump
python ./distance.py --compare -g 10 -d 98 --m1 300 --m2 900 --r2 620

python ./fail_attack.py -n 728 -m 1200 -g 10 -m1 300 -d 120 -d1 302 --C2-type orthogonal --dump
python ./distance.py --compare -g 10 -d 120 --m1 300 --m2 900 --r2 598

python ./fail_attack.py -n 700 -m 1200 -g 10 -m1 300 -d 70 -d1 280 --C2-type orthogonal --dump
python ./distance.py --compare -g 10 -d 70 --m1 300 --m2 900 --r2 620

python ./fail_attack.py -n 630 -m 1200 -g 10 -m1 300 -d 70 -d1 350 --C2-type orthogonal --AB-type from_C --low_weight_H_s --column-red zero --read log/ccebb2e7-d1af-4ebb-abcf-005a99d14357.json # succeeded

python ./fail_attack.py -n 630 -m 1200 -g 10 -m1 300 -d 70 -d1 350 --C2-type orthogonal --AB-type independent --low_weight_H_s --d_red 100 --dump # almost succeeded

python ./fail_attack.py -n 630 -m 1200 -g 10 -m1 300 -d 70 -d1 350 --C2-type orthogonal --AB-type independent --low_weight_H_s --d_red 100 --dump # change the AB blocks, so that it is not fully independent. almost succeeded

python ./fail_attack.py -n 630 -m 1200 -g 10 -m1 300 -d 70 -d1 350 --C2-type orthogonal --AB-type independent --low_weight_H_s --d_red 100 --read log/16a5d596-353b-4f7d-9600-b9850438710c.json --p 0.35 --E 1000 --fine-tune # found 882 redundant rows

python ./fail_attack.py -n 630 -m 1200 -g 10 -m1 300 -d 70 -d1 350 --C2-type orthogonal --AB-type independent --low_weight_H_s --d_red 100 --read log/5364f16f-b670-4fe5-99e3-eb1bd51a2939.json --p 0.35 --E 700 --fine-tune # found 730 redundant rows

python ./fail_attack.py -n 630 -m 1200 -g 10 -m1 300 -d 143 -d1 423 --C2-type orthogonal --AB-type from_C # failed

python ./fail_attack.py -n 650 -m 1200 -g 10 -m1 300 -d 143 -d1 403 --C2-type orthogonal --AB-type from_C --column-red FD_col --dump # failed

python ./fail_attack.py -n 660 -m 1200 -g 10 -m1 300 -d 143 -d1 393 --C2-type orthogonal --AB-type from_C --column-red FD_col --dump # succeeded

python ./fail_attack.py -n 660 -m 1200 -g 10 -m1 300 -d 143 -d1 393 --C2-type orthogonal --AB-type zero --d_red 20 --dump # failed, but H not full rank

python ./fail_attack.py -n 660 -m 1200 -g 10 -m1 300 -d 143 -d1 393 --C2-type orthogonal --AB-type from_C --d_red 20 --dump # failed, but H not full rank

python ./fail_attack.py -n 660 -m 1200 -g 10 -m1 300 -d 143 -d1 393 --C2-type orthogonal --AB-type independent --d_red 20 --dump # succeeded


python ./fail_attack.py -n 660 -m 1200 -g 10 -m1 300 -d 143 -d1 393 --C2-type orthogonal --AB-type zero --dump # log/1c5dd280-8f0a-456d-a286-227ae72dd690.json

python ./fail_attack.py -n 660 -m 1200 -g 10 -m1 300 -d 143 -d1 393 --C2-type orthogonal --AB-type zero --read log/1c5dd280-8f0a-456d-a286-227ae72dd690.json --p 0.38 --E 1200 --fine-tune # found 396 redundant rows

python ./fail_attack.py -n 660 -m 1200 -g 10 -m1 300 -d 140 -d1 390 --C2-type orthogonal --AB-type zero --concat_D --dump # log/5ecf7fda-fbd7-4a6c-93a9-a19601f287a3.json

python ./fail_attack.py -n 700 -m 1200 -g 10 -m1 300 -d 140 -d1 350 --C2-type orthogonal --AB-type zero --concat_D --dump # log/4a6f4922-33e3-4f8b-b683-bc9553a1cf88.json

python ./fail_attack.py -n 700 -m 1200 -g 10 -m1 300 -d 140 -d1 350 --C2-type orthogonal --AB-type zero --concat_D --read log/4a6f4922-33e3-4f8b-b683-bc9553a1cf88.json --p 0.33 --E 500 --fine-tune

python ./fail_attack.py -n 700 -m 1200 -g 10 -m1 300 -d 140 -d1 350 --C2-type orthogonal --AB-type independent --concat_D --dump # log/0241a455-4000-42f8-b613-a1d6bcd58729.json, found 844 redundant rows

python ./fail_attack.py -n 700 -m 1200 -g 10 -m1 300 -d 140 -d1 350 --AB-type zero --concat_D --concat_C1 -m0 60 -d0 28 --dump # log/f5181ec8-2123-45a9-887e-5cc2841be209.json

python ./fail_attack.py -n 700 -m 1200 -g 10 -m1 300 -d 140 -d1 350 --AB-type zero --concat_D --concat_C1 -m0 60 -d0 28 --read log/f5181ec8-2123-45a9-887e-5cc2841be209.json --p 0.2 --E 2000 --fine-tune

python ./fail_attack.py -n 750 -m 1200 -g 10 -m1 300 -d 140 -d1 300 --AB-type zero --concat_D --concat_C1 -m0 60 -d0 28 --dump # log/ce76ddba-dde7-461b-b411-c2c14a8d1acb.json

python ./fail_attack.py -n 750 -m 1200 -g 10 -m1 300 -d 140 -d1 300 --AB-type zero --concat_D --concat_C1 -m0 60 -d0 28 --read log/ce76ddba-dde7-461b-b411-c2c14a8d1acb.json --p 0.2 --E 2000 --fine-tune

python ./fail_attack.py -n 750 -m 1200 -g 10 -m1 300 -d 140 -d1 300 --AB-type zero --concat_D --concat_C1 -m0 60 -d0 28 --read log/ce76ddba-dde7-461b-b411-c2c14a8d1acb.json --p 0.3 --E 400 --fine-tune

python ./fail_attack.py -n 750 -m 1200 -g 10 -m1 300 -d 140 -d1 0 --AB-type concat --concat_D -m0 30 -d0 14 --dump # log/79811830-56bc-4cd3-bcdd-ed943acdd5b7.json

python ./fail_attack.py -n 750 -m 1200 -g 10 -m1 300 -d 140 -d1 0 --AB-type concat --concat_D -m0 30 -d0 14 --read log/79811830-56bc-4cd3-bcdd-ed943acdd5b7.json --p 0.3 --E 200 --fine-tune 