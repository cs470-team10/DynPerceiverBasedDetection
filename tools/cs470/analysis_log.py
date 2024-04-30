# 정지용 작성
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(
        'Log Analysis Script', add_help=False)
    parser.add_argument('--src', type=str, default='', help='Baseline log, multiple logs are splited as ---')
    parser.add_argument('--dest', type=str, default='', help='Target log, multiple logs are splited as ---')
    parser.add_argument('--out', type=str, default='', help='(Optional) Analysis output log')
    parser.add_argument('--src_type', type=str, default='log', help='Src file type', choices=['log', 'json'])
    parser.add_argument('--dest_type', type=str, default='log', help='Dest file type', choices=['log', 'json'])
    parser.add_argument('--num_epochs', type=int, default=12, help='Number of epochs')
    return parser

def find_in_line(line: str, num_epochs:int = 12, file_type:str = "log"):
    if file_type == "log":
        for i in range(1, num_epochs + 1):
            query = f"INFO - Epoch(val) [{str(i)}][5000/5000]    coco/bbox_mAP: "
            if query in line:
                accuarcy = float(line.split(query)[1].split(" ")[0])
                return True, i - 1, accuarcy
    else:
        for i in range(1, num_epochs + 1):
            query = 'mode": "val", "epoch": ' + str(i) +', "iter": 7330, "'
            if query in line:
                accuarcy = float(line.split('"bbox_mAP_copypaste": "')[1].split(" ")[0])
                return True, i - 1, accuarcy
    return False, -1, -1.0

def find_in_file(filenames: str, num_epochs: int = 12, file_type: str = "log"):
    accuarcy_array = [-1.0 for i in range(num_epochs)]
    for filename in filenames.split("---"):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                result, epoch, accuarcy = find_in_line(line, 12, file_type)
                if result is True:
                    accuarcy_array[epoch] = accuarcy
    return accuarcy_array

def main(args):
    src, dest, out, src_type, dest_type, num_epochs = str(args.src), str(args.dest), str(args.out), str(args.src_type), str(args.dest_type), int(args.num_epochs)
    if src_type == 'log':
        assert src.endswith(".log") or src.endswith(".out"), "src file은 log 또는 out 형식의 파일이어야 합니다."
    else:
        assert src.endswith(".json"), "src file은 json 형식의 파일이어야 합니다."
    if dest_type == 'log':
        assert dest.endswith(".log") or dest.endswith(".out"), "dest file은 log 또는 out 형식의 파일이어야 합니다."
    else:
        assert dest.endswith(".json"), "dest file은 json 형식의 파일이어야 합니다."
    src_accuarcy = find_in_file(src, num_epochs, src_type)
    dest_accuarcy = find_in_file(dest, num_epochs, dest_type)
    for i in range(num_epochs):
        if (src_accuarcy[i] == -1.0 or dest_accuarcy[i] == -1.0):
            continue
        difference = dest_accuarcy[i] - src_accuarcy[i]
        print(f"Epoch {str(i + 1)}: " + '{:.1%} ('.format(dest_accuarcy[i]) + ('+' if difference >= 0 else '-') + '{:.1%})'.format(difference if difference >= 0 else -difference))
        print("\n")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Log Analysis Script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
