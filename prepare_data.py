import os
import json

from parlai.core.params import ParlaiParser
from parlai.agents.fixed_response.fixed_response import FixedResponseAgent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger


def prepare_dataset(opt):
    task = opt['task']
    dt = opt['datatype'].split(':')[0]
    
    conv_idx, utter_idx = 0, 0

    agent = FixedResponseAgent(opt)
    world = create_task(opt, agent)
    log_timer = TimeLogger()

    file_root_dir = os.path.join(opt['prepare_data_dir'], task)
    os.makedirs(file_root_dir, exist_ok=True)
    
    cnt = 0
    
    f = open(os.path.join(file_root_dir, f'{dt}.jsonl'), 'w', encoding='utf-8')

    while not world.epoch_done():
        cnt += 1
        world.parley()

        context = world.acts[0].get('text')
        response = world.acts[0].get('labels', world.acts[0].get('eval_labels'))[0]
        episode_done = world.acts[0].get('episode_done')
        emotion = world.acts[0].get('emotion')
        situation = world.acts[0].get('situation')
        
        obj1 = json.dumps({'conv_idx': conv_idx, 'utter_idx': utter_idx, 'utterance': context, 'emotion': emotion, 'situation': situation})
        f.write(obj1 + '\n')
        utter_idx += 1
        obj2 = json.dumps({'conv_idx': conv_idx, 'utter_idx': utter_idx, 'utterance': response, 'emotion': emotion, 'situation': situation})
        f.write(obj2 + '\n')
        utter_idx += 1

        if episode_done:
            conv_idx += 1
            utter_idx = 0

        if log_timer.time() > opt['log_every_n_secs']:
            text, _log = log_timer.log(world.total_parleys, world.num_examples())
            print(text)
    
    f.close()

if __name__ == '__main__':
    parser = ParlaiParser()
    parser.add_argument(
        "--prepare-data-dir",
        default=None,
        type=str,
        help='Save to the prepared dataset'
    )
    opt = parser.parse_args()
    
    # Common options
    opt['log_every_n_secs'] = 2
    opt['fixed_response'] = None

    # prepare dataset
    prepare_dataset(opt)