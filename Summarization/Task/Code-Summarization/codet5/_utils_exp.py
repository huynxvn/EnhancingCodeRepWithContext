import json

def add_lang_by_task(target_str, task, sub_task):
    if task == 'summarize':
        target_str = '<en> ' + target_str
    elif task == 'refine':
        target_str = '<java> ' + target_str
    elif task == 'translate':
        if sub_task == 'java-cs':
            target_str = '<c_sharp> ' + target_str
        else:
            target_str = '<java> ' + target_str
    elif task == 'concode':
        target_str = '<java> ' + target_str
    elif task == 'defect':
        target_str = target_str
    return target_str


def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item

    # Add task prefix if specified (for 't5' and 'codet5' models)
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        if args.sub_task != 'none':
            source_str = "{} {}: {}".format(args.task, args.sub_task, example.source)
        else:
            source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source

    # Replace unknown tokens and make sure special tokens are recognized
    source_str = source_str.replace('</s>', '<unk>')

    # Tokenize without truncating to check the original token count
    tokenized_source_ids = tokenizer.encode(source_str, add_special_tokens=True)

    # Flag whether the input exceeds the token limit before truncation
    exceeds_token_limit = len(tokenized_source_ids) > args.max_source_length

    # Encode the source string using the tokenizer with special tokens support, truncating if necessary
    source_ids = tokenizer.encode(
        source_str,
        max_length=args.max_source_length,
        padding='max_length',
        truncation=True
    )

    # Truncate manually while ensuring space for the ending token
    if len(source_ids) > args.max_source_length:
        source_ids = source_ids[:args.max_source_length - 1]

    # Check if the eos token is present in the truncated sequence
    eos_found = tokenizer.eos_token_id in source_ids

    # Add padding or eos token accordingly
    if eos_found:
        # If the eos token is already present, ensure the sequence is padded to max length
        if len(source_ids) < args.max_source_length:
            source_ids.append(0)  # Append a padding token
    else:
        # If the eos token is not present, add it to the end of the sequence
        source_ids.append(tokenizer.eos_token_id)

    # Ensure the length is equal to max_source_length
    assert len(source_ids) == args.max_source_length, f"source_ids length {len(source_ids)} does not match max_source_length {args.max_source_length}"

    # Ensure eos_token_id is used correctly (only one instance)
    eos_count = source_ids.count(tokenizer.eos_token_id)
    assert eos_count <= 1, f"More than one eos_token_id found in source_ids: {source_ids}"

    # Prepare target string if the stage is not 'test'
    if stage == 'test':
        target_ids = []
    else:
        target_str = example.target
        if args.add_lang_ids:
            # Add language-specific information to the target string
            target_str = add_lang_by_task(target_str, args.task, args.sub_task)

        # Handle defect or clone tasks with specific labels
        if args.task in ['defect', 'clone']:
            if target_str == 0:
                target_str = 'false'
            elif target_str == 1:
                target_str = 'true'
            else:
                raise NameError

        # Replace unknown tokens and encode the target string
        target_str = target_str.replace('</s>', '<unk>')
        target_ids = tokenizer.encode(
            target_str,
            max_length=args.max_target_length,
            padding='max_length',
            truncation=True
        )

        if tokenizer.eos_token_id is not None:
            assert target_ids.count(tokenizer.eos_token_id) <= 1

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
        url=example.url,
        exceeds_token_limit=int(exceeds_token_limit)  # Convert flag to an integer for Tensor conversion
    )

# def convert_clone_examples_to_features(item):
#     example, example_index, tokenizer, args = item
#     if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
#         source_str = "{}: {}".format(args.task, example.source)
#         target_str = "{}: {}".format(args.task, example.target)
#     else:
#         source_str = example.source
#         target_str = example.target
#     code1 = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
#     code2 = tokenizer.encode(target_str, max_length=args.max_source_length, padding='max_length', truncation=True)
#     source_ids = code1 + code2
#     return CloneInputFeatures(example_index, source_ids, example.label, example.url1, example.url2)

def convert_clone_examples_to_features(item):
    example, example_index, tokenizer, args = item

    # Add task prefix if specified (for 't5' and 'codet5' models)
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
        target_str = "{}: {}".format(args.task, example.target)
    else:
        source_str = example.source
        target_str = example.target

    # Replace unknown tokens and encode source and target strings
    source_str = source_str.replace('</s>', '<unk>')
    target_str = target_str.replace('</s>', '<unk>')

    code1 = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    code2 = tokenizer.encode(target_str, max_length=args.max_source_length, padding='max_length', truncation=True)

    # Ensure eos_token_id is used correctly (only one instance)
    if tokenizer.eos_token_id is not None:
        assert code1.count(tokenizer.eos_token_id) <= 1
        assert code2.count(tokenizer.eos_token_id) <= 1

    source_ids = code1 + code2

    return CloneInputFeatures(example_index, source_ids, example.label, example.url1, example.url2)


# def convert_defect_examples_to_features(item):
#     example, example_index, tokenizer, args = item
#     if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
#         source_str = "{}: {}".format(args.task, example.source)
#     else:
#         source_str = example.source
#     code = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
#     return DefectInputFeatures(example_index, code, example.target)

def convert_clone_examples_to_features(item):
    example, example_index, tokenizer, args = item

    # Add task prefix if specified (for 't5' and 'codet5' models)
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
        target_str = "{}: {}".format(args.task, example.target)
    else:
        source_str = example.source
        target_str = example.target

    # Replace unknown tokens and encode source and target strings
    source_str = source_str.replace('</s>', '<unk>')
    target_str = target_str.replace('</s>', '<unk>')

    code1 = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    code2 = tokenizer.encode(target_str, max_length=args.max_source_length, padding='max_length', truncation=True)

    # Ensure eos_token_id is used correctly (only one instance)
    if tokenizer.eos_token_id is not None:
        assert code1.count(tokenizer.eos_token_id) <= 1
        assert code2.count(tokenizer.eos_token_id) <= 1

    source_ids = code1 + code2

    return CloneInputFeatures(example_index, source_ids, example.label, example.url1, example.url2)



class CloneInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label,
                 url1,
                 url2
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label
        self.url1 = url1
        self.url2 = url2


class DefectInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None,
                 exceeds_token_limit=False  # store the flag
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url
        self.exceeds_token_limit = exceeds_token_limit



class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task


class CloneExample(object):
    """A single training/test example."""

    def __init__(self,
                 code1,
                 code2,
                 label,
                 url1,
                 url2
                 ):
        self.source = code1
        self.target = code2
        self.label = label
        self.url1 = url1
        self.url2 = url2


def read_translate_examples(filename, data_num, args):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            src = line1.strip()
            trg = line2.strip()
            examples.append(
                Example(
                    idx=idx,
                    source=src,
                    target=trg,
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_refine_examples(filename, data_num, args):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0

    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target=line2.strip(),
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_concode_examples(filename, data_num, args):
    """Read examples from filename."""
    examples = []

    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=x["code"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_summarize_examples(filename, data_num, args):
    """Read examples from filename."""
    examples = []    
    with open(filename, encoding="utf-8") as f:        
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx

            code = ''
            if args.exp == "baseline":
                source_code = ' '.join(js['code_tokens']).replace('\n', ' ')
                combined_code = (
                    f"<code> {source_code} "
                )
            
            elif args.exp == "code_vh":                
                # code + version_history
                source_code = ' '.join(js['code_tokens']).replace('\n', ' ')
                version_history = '<version> '.join([' '.join(item['commit_source_code_tokens']).replace('\n', ' ') for item in js['version_history'][1:]])
                # Combine features using the special tokens
                combined_code = (
                    f"<code> {source_code} "
                    f"<history> {version_history} "
                )
            
            elif args.exp == "code_cg":
                # code + callgraph
                source_code = ' '.join(js['code_tokens']).replace('\n', ' ')
                caller_code = ' '.join(js['caller_context_tokens']).replace('\n', ' ')
                callee_code = ' '.join(js['callee_context_tokens']).replace('\n', ' ')
                # Combine features using the special tokens
                combined_code = (
                    f"<code> {source_code} "
                    f"<caller> {caller_code} "
                    f"<callee> {callee_code} "
                )

            elif args.exp == "code_vh_nod":                
                # code + version_history + num_of_days                
                source_code = ' '.join(js['code_tokens']).replace('\n', ' ')
                version_history = '<version> '.join([' '.join(item['commit_source_code_tokens']).replace('\n', ' ') for item in js['version_history'][1:]])
                num_of_days = ' '.join(js['num_of_days_tokens'])
                # Combine features using the special tokens
                combined_code = (
                    f"<code> {source_code} "
                    f"<history> {version_history} "
                    f"<days> {num_of_days} "
                )
            
            elif args.exp == "code_vh_cg":
                # code + version_history + callgraph
                source_code = ' '.join(js['code_tokens']).replace('\n', ' ')
                version_history = '<version> '.join([' '.join(item['commit_source_code_tokens']).replace('\n', ' ') for item in js['version_history'][1:]])                
                caller_code = ' '.join(js['caller_context_tokens']).replace('\n', ' ')
                callee_code = ' '.join(js['callee_context_tokens']).replace('\n', ' ')
                # Combine features using the special tokens
                combined_code = (
                    f"<code> {source_code} "
                    f"<history> {version_history} "
                    f"<caller> {caller_code} "
                    f"<callee> {callee_code} "
                )

            elif args.exp == "code_cg_vh":
                # code + version_history + callgraph
                source_code = ' '.join(js['code_tokens']).replace('\n', ' ')                
                caller_code = ' '.join(js['caller_context_tokens']).replace('\n', ' ')
                callee_code = ' '.join(js['callee_context_tokens']).replace('\n', ' ')
                version_history = '<version> '.join([' '.join(item['commit_source_code_tokens']).replace('\n', ' ') for item in js['version_history'][1:]])
                
                # Combine features using the special tokens
                combined_code = (
                    f"<code> {source_code} "
                    f"<caller> {caller_code} "
                    f"<callee> {callee_code} "
                    f"<history> {version_history} "                    
                )
            
            elif args.exp == "code_vh_cg_nod":
                # code + version_history + callgraph + num_of_days
                source_code = ' '.join(js['code_tokens']).replace('\n', ' ')
                version_history = '<version> '.join([' '.join(item['commit_source_code_tokens']).replace('\n', ' ') for item in js['version_history'][1:]])
                caller_code = ' '.join(js['caller_context_tokens']).replace('\n', ' ')
                callee_code = ' '.join(js['callee_context_tokens']).replace('\n', ' ')
                num_of_days = ' '.join(js['num_of_days_tokens'])
                # Combine features using the special tokens
                combined_code = (
                    f"<code> {source_code} "
                    f"<history> {version_history} "
                    f"<caller> {caller_code} "
                    f"<callee> {callee_code} "
                    f"<days> {num_of_days} "
                )
            
            elif args.exp == "code_cg_vh_nod":
                # code + version_history + callgraph + num_of_days
                source_code = ' '.join(js['code_tokens']).replace('\n', ' ')
                caller_code = ' '.join(js['caller_context_tokens']).replace('\n', ' ')
                callee_code = ' '.join(js['callee_context_tokens']).replace('\n', ' ')
                version_history = '<version> '.join([' '.join(item['commit_source_code_tokens']).replace('\n', ' ') for item in js['version_history'][1:]])
                num_of_days = ' '.join(js['num_of_days_tokens'])
                # Combine features using the special tokens
                combined_code = (
                    f"<code> {source_code} "
                    f"<caller> {caller_code} "
                    f"<callee> {callee_code} "
                    f"<history> {version_history} "
                    f"<days> {num_of_days} "
                )

            else:
                print("========>", args.exp, "null or unknown experiment params!!!")
                breakpoint()

            code = ' '.join(combined_code.split())            
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples

def read_defect_examples(filename, data_num, args):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)

            code = ' '.join(js['func'].split())
            examples.append(
                Example(
                    idx=js['idx'],
                    source=code,
                    target=js['target']
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_clone_examples(filename, data_num, args):
    """Read examples from filename."""
    index_filename = filename
    url_to_code = {}
    with open('/'.join(index_filename.split('/')[:-1]) + '/{}/data.jsonl'.format(args.test_type)) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())
            url_to_code[js['idx']] = code

    data = []
    with open(index_filename) as f:
        idx = 0
        for line in f:
            line = line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label == '0':
                label = 0
            else:
                label = 1
            data.append(CloneExample(url_to_code[url1], url_to_code[url2], label, url1, url2))
            idx += 1
            if idx == data_num:
                break
    return data
