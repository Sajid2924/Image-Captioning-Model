# Count Flickr8k pairs (subtract 1 for header line)
echo "=== Flickr8k ===" && echo "Pairs (lines-1):" && wc -l data/captions.txt && echo "Unique images:" && tail -n +2 data/captions.txt | cut -d',' -f1 | sort -u | wc -l

# Count COCO 2014 train
echo "=== COCO 2014 train ===" && python3 -c "
import json
with open('data/annotations_trainval2014/captions_train2014.json') as f:
    d = json.load(f)
print('Images:', len(d['images']))
print('Captions:', len(d['annotations']))
"

# Count COCO 2017 val
echo "=== COCO 2017 val ===" && python3 -c "
import json
with open('data/annotations_trainval2017/captions_val2017.json') as f:
    d = json.load(f)
print('Images:', len(d['images']))
print('Captions:', len(d['annotations']))
"