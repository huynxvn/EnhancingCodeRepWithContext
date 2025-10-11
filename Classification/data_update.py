import json
import os
import sys

import numpy as np
import pandas as pd

def fn_search_dict_by_params(data_list, project, file, method):
    for data_dict in data_list:
        if data_dict['project'] == project \
            and "/".join(data_dict['file'].split('/')[2:]) == file \
            and data_dict['method'] == method:
            return data_dict
    return None  # Return None if no matching dictionary is found

def get_context(callgraph_dir, project_name, class_name, method_name):
    with open(callgraph_dir + '/'+project_name+'.cg', 'r', encoding='utf-8') as cg_file:
        call_graph = cg_file.readlines()
    record = []
    for line in call_graph:
        if line.startswith('M') and class_name in line and method_name in line:
            line_list = line.split(' ')
            if class_name in line_list[0] and method_name in line_list[0]:
                if class_name not in line_list[1] or method_name not in line_list[1]:
                    record.append((1, line_list[1].strip()))
            elif class_name not in line_list[0] or method_name not in line_list[0]:
                record.append((0, line_list[0].strip()))
    record = list(set(record))
    return record

def parse_path(project: str, cg_node: str):
    root_dict = {'caffeine': 'caffeine/caffeine/src/main/java',
                 'checkstyle': 'checkstyle/src/main/java',
                 'commons-collections': 'commons-collections/src/main/java',
                 'commons-lang': 'commons-lang/src/main/java',
                 'commons-math': 'commons-math/src/main/java',
                 'deeplearning4j': 'deeplearning4j/nd4j/nd4j-common/src/main/java',
                 'eclipse.jdt.core': 'eclipse.jdt.core/org.eclipse.jdt.apt.core/src',
                 'freemind': 'freemind/freemind/freemind',
                 'guava': 'guava/guava/src',
                 'openjdk11': 'openjdk11/src',
                 'trove': 'trove/core/src/main/java'
                 }
    root = root_dict[project]
    file_raw_path = '/'.join(cg_node.strip('(M)').strip('M:').split(':')[0].split('.')) + '.java'
    file_path = root + '/' + file_raw_path
    method = cg_node.strip('(M)').strip('M:').split(':')[1].split('(')[0]
    return file_path, method

def extract_fragment(file_path, method_name):
    if not os.path.exists(file_path):
        flag = False
        file_name = file_path.split('/')[-1]
        # You need to modify the search range if you do not use the default repos path.
        for root, dirs, files in os.walk('/'.join(file_path.split('/')[:5])):
            if file_name in files:
                file_path = root + '/' + file_name
                flag = True
                break
        if not flag:
            # print('路径不存在：', file_path)
            return
    with open(file_path, encoding='utf-8') as file:
        file_lines = file.readlines()
        num_lines = len(file_lines)
    ret = []
    for i, line in enumerate(file_lines):
        if line.find(method_name) != -1 \
                and ('static' in line
                     or 'final' in line
                     or 'public' in line
                     or 'protected' in line
                     or 'private' in line
                     or 'default' in line
                     or 'void' in line
                     or 'boolean' in line
                     or 'int' in line
                     or 'long' in line
                     or 'byte' in line
                     or line.strip().startswith('<T>')
                     or line.strip().startswith('/*non-public*/')
                     or line.strip()[0].isupper()) \
                and not (line.strip().startswith('/**')
                         or line.strip().startswith('*')
                         or line.strip().startswith('*/')
                         or line.strip().startswith('//')) \
                and ' new ' not in line \
                and not line.strip().endswith(';')\
                and line.find('(') > line.find(method_name):
            index = i - 1
            while index >= 0 and ('*' in file_lines[index] or '@' in file_lines[index]):
                index -= 1
            start_id = index + 1
            index = i
            stack = ['']
            while index < num_lines and len(stack) != 0:
                for char in file_lines[index]:
                    if char == '{':
                        if stack[0] == '':
                            stack[0] = '{'
                        else:
                            stack.append('{')
                    elif char == '}':
                        stack.pop(-1)
                        if len(stack) == 0:
                            break
                index += 1
            end_id = index
            ret_ins = ''.join(file_lines[start_id:end_id])
            ret.append(ret_ins)
    return '\n'.join(ret)


class Pipeline:
    def __init__(self, source_data_path, repos_dir, callgraph_dir, version_history_dir, save_dir, save_filename):
        self.source_data_path = source_data_path
        self.repos_dir = repos_dir
        self.callgraph_dir = callgraph_dir
        self.version_history_dir = version_history_dir
        self.save_dir = save_dir
        self.save_filename = save_filename
        self.dataset = None
        self.ds_version_history = None  

    def extract_code(self):
        # original SeSaMe dataset
        with open(self.source_data_path, 'r', encoding='utf-8') as source_data_file:
            self.dataset = json.load(source_data_file)
            print("origin:", len(self.dataset))
            source_data_file.close()
        
        # version history dataset        
        with open(self.version_history_dir, 'r') as json_file:
            self.ds_version_history = json.load(json_file)
            json_file.close()

        new_data = [] # output
        
        length = len(self.dataset)
        idx = 0
        for data_ins in self.dataset:
            a_dict_record = {}
            new_sample_first = {}
            new_sample_second = {}

            # sample_first = sample['first']
            # sample_second = sample['second']
            # new_sample_first = sample_first

            code1 = data_ins['first']
            code2 = data_ins['second']

            new_sample_first = code1

            a_context = fn_search_dict_by_params(self.ds_version_history,
                code1['project'],
                code1['file'],
                code1['method'],
                ) 
            if a_context is not None:
                new_sample_first['code'] = a_context['version_history'][0]['commit_source_code']
                # new_sample_first['context'] = ""
                
                new_sample_first['uniqueid'] = a_context['uniqueid']
                new_sample_first['version_history_context'] = a_context['version_history']
                new_sample_first['number_of_versions'] = a_context['number_of_versions']
                new_sample_first['days_to_exist'] = a_context['days_to_exist']
            else:
                print("Warning: sample 1st's version history not found", code1)
            
            
            # code1 - for callgraph reproducibility
            file_path = '/'.join([self.repos_dir, code1['project'], code1['file']])
            method_name = code1['method'].split('(')[0].split('.')[-1]
            extracted_code1 = extract_fragment(file_path, method_name)
            extracted_code1 = ""
            code1['callgraph_code_v1'] = extracted_code1
            # print(code1['code'])

            new_sample_second = code2

            a_context = fn_search_dict_by_params(self.ds_version_history,
                code2['project'],
                code2['file'],
                code2['method'],
                ) 
            if a_context is not None:
                new_sample_second['code'] = a_context['version_history'][0]['commit_source_code']
                # new_sample_second['context'] = ""

                new_sample_second['uniqueid'] = a_context['uniqueid']
                new_sample_second['version_history_context'] = a_context['version_history']
                new_sample_second['number_of_versions'] = a_context['number_of_versions']
                new_sample_second['days_to_exist'] = a_context['days_to_exist']
            else:
                print("Warning: sample 2nd's version history not found", code2)

            # # code2 - for callgraph reproducibility
            file_path = '/'.join([self.repos_dir, code2['project'], code2['file']])
            method_name = code2['method'].split('(')[0].split('.')[-1]
            extracted_code2 = extract_fragment(file_path, method_name)
            code2['callgraph_code_v1'] = extracted_code2
            # print(code2['code'])

            a_dict_record = data_ins
            # a_dict_record['first_version_history_context'] = new_sample_first['version_history_context']
            # a_dict_record['second_version_history_context'] = new_sample_second['version_history_context']
            new_data.append(a_dict_record)
            
            idx += 1
            print('\r', end='')
            print(f'{idx}/{length}', end='')
            sys.stdout.flush()
        print()
        return self.dataset

    def extract_context(self):
        length = len(self.dataset)
        new_ds = []

        count_null_cg = 0        
        count_null_cg_v1 = 0
        count_null_bothcg = 0
        
        for idx, sample in enumerate(self.dataset):
            sample_first = sample['first']
            sample_second = sample['second']

            project_name = sample_first['project']
            class_name = sample_first['method'].split('.')[0]
            method_name = sample_first['method'].split('.')[-1].split('(')[0]
            first_context_pointers = get_context(self.callgraph_dir, project_name, class_name, method_name)
            
            first_context_res_v1 = []
            first_context_res = []

            for relation, node in first_context_pointers:
                context_path, context_name = parse_path(sample_first['project'], node)
                
                # first context - for callgraph reproducibility
                context_code_v1 = extract_fragment(self.repos_dir + '/' + context_path, context_name)
                if context_code_v1:
                    first_context_res_v1.append((relation, node, context_code_v1))
                
                # callgraph v2 - extracted from version history
                context_code = sample_first['code']
                if context_code:
                    first_context_res.append((relation, node, context_code))

            project_name = sample_second['project']
            class_name = sample_second['method'].split('.')[0]
            method_name = sample_second['method'].split('.')[-1].split('(')[0]
            second_context_pointers = get_context(self.callgraph_dir, project_name, class_name, method_name)
            
            second_context_res_v1 = []
            second_context_res = []
            
            for relation, node in second_context_pointers:
                context_path, context_name = parse_path(sample_second['project'], node)

                # second context - for callgraph reproducibility
                context_code_v1 = extract_fragment(self.repos_dir + '/' + context_path, context_name)
                if context_code_v1:
                    second_context_res_v1.append((relation, node, context_code_v1))
                
                # callgraph v2 - extracted from version history
                context_code = sample_second['code']
                if context_code:
                    second_context_res.append((relation, node, context_code))
            
            # # We only need the data with at least one snippet of context code.
            # if len(first_context_res) != 0 or len(second_context_res) != 0:
            #     sample_first['context'] = first_context_res
            #     sample_second['context'] = second_context_res
            #     new_ds.append(sample)


            # # for the case of callgraph extracted from version history
            # if len(first_context_res) != 0 or len(second_context_res) != 0:
            #     sample['callgraph_available'] = 1                
            #     sample_first['callgraph_context'] = first_context_res
            #     sample_second['callgraph_context'] = second_context_res
            #     new_ds.append(sample)                
            # else:                
            #     count_null_cg += 1
            #     sample['callgraph_available'] = 0

            # We only need the data with at least one snippet of context code.            
            # if len(first_context_res_v1) != 0 or len(second_context_res_v1) != 0:

            # if len(first_context_res_v1) != 0 or len(second_context_res_v1) != 0:
            #     sample['callgraph_available_v1'] = 1

            #     # callgraph v2 - extracted from version history
            #     sample_first['callgraph_context'] = first_context_res
            #     sample_second['callgraph_context'] = second_context_res

            #     # callgraph v1 - reproducibility
            #     sample_first['callgraph_context_v1'] = first_context_res_v1
            #     sample_second['callgraph_context_v1'] = second_context_res_v1

            #     if len(first_context_res) == 0 and len(second_context_res) == 0:
            #         print("have v1 callgraph but not have verhis callgraph", sample_first['uniqueid'], sample_second['uniqueid'])

            #     new_ds.append(sample)
            # else:
            #     count_null_cg += 1
            #     sample['callgraph_available_v1'] = 0


            chk_bothcg_availbale = True

            if len(first_context_res) == 0 and len(second_context_res) == 0:
                sample['callgraph_available'] = 0
                count_null_cg += 1
                chk_bothcg_availbale = False
            else:
                sample['callgraph_available'] = 1
            # callgraph v2 - extracted from version history
            sample_first['callgraph_context'] = first_context_res
            sample_second['callgraph_context'] = second_context_res

            if len(first_context_res_v1) == 0 and len(second_context_res_v1) == 0:                
                sample['callgraph_available_v1'] = 0
                count_null_cg_v1 += 1
                chk_bothcg_availbale = False
            else:
                sample['callgraph_available_v1'] = 1
            # callgraph v1 - reproducibility
            sample_first['callgraph_context_v1'] = first_context_res_v1
            sample_second['callgraph_context_v1'] = second_context_res_v1


            if chk_bothcg_availbale == False:
                count_null_bothcg += 1

            new_ds.append(sample)
                            
            print('\r', end='')
            print(f'{idx + 1}/{length}', end='')
            sys.stdout.flush()
        
        print()
        self.dataset = new_ds
        print("null callgraph v1/total:", count_null_cg_v1,"/",length)
        print("null callgraph/total:", count_null_cg,"/",length)
        print("null both callgraph/total:", count_null_bothcg,"/",length)

        print("remaining record:", len(self.dataset))

        return self.dataset
    
    def exclude_javalang_failed_cases(self):
        # lst_err = [147, 308, 882, 1218]
        # 27894, 30344, 24082, 17670
        lst_err = [147, 308, 882, 1218, 927, 1196, 1539, 816]
        # lst_err = []
        final_data = []
        total = len(self.dataset)
        for row in self.dataset:
            if row['first']['uniqueid'] in lst_err or row['second']['uniqueid'] in lst_err:
                print(row['first']['uniqueid'], row['second']['uniqueid'])
            else:
                final_data.append(row)
        self.dataset = final_data.copy()
        print("remaining/total", len(final_data), "/", total)
        return self.dataset

    def save(self):
        with open(self.save_dir + '/' + self.save_filename, 'w', encoding='utf-8') as save_file:
            json.dump(self.dataset, save_file, indent=4, ensure_ascii=False)
        return True
    
    def run(self):
        print('Start extracting the code of target method.')
        self.extract_code()
        print('Extraction ends.')
        print('Start extracting the code of context.')
        self.extract_context()
        print('Extraction ends.')
        print('Excluding javalang failed cases.')
        self.exclude_javalang_failed_cases()
        print('Excluding process completed.')
        self.save()
        print('The dataset with context has been saved.')

if __name__ == '__main__':
    # SOURCE_DATA_PATH = './sesame/dataset.json'
    SOURCE_DATA_PATH = 'sesame/sesame-origin/dataset.json'
    REPOS_DIR = './sesame/src/repos'    
    CALLGRAPH_DIR = './callgraphs'
    VERSION_HISTORY_PATH = 'mining/output/version_history.json'
    SAVE_DIR = './data'
    SAVE_FILE_NAME = 'SeSaMe_VersionHistory_Callgraph.vFinal.json'

    ppl = Pipeline(SOURCE_DATA_PATH,  REPOS_DIR, CALLGRAPH_DIR, VERSION_HISTORY_PATH, SAVE_DIR, SAVE_FILE_NAME)
    ppl.run()