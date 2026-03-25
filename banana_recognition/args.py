import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Model training options")

    parser.add_argument('--backbone', type=str, default='fasterrcnn_mobilenet_v3',
                        choices=['fasterrcnn_resnet50_fpn','fasterrcnn_mobilenet_v3']) # previously default='fasterrcnn_resnet50_fpn' 

    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=512)
   
    
    parser.add_argument('--csv_dir', type=str, default='banana_recognition/data/CSVs')
    parser.add_argument('--out_dir', type=str, default='./sessions')

    parser.add_argument('--batch_size', type=int, default=8, choices=[8,16,32,64]) # previously 32
    parser.add_argument('--epochs', type=int, default=25) # previously 100
    parser.add_argument('--lr', type=float, default=0.00005) # learning rate, adjust for learning curve. - previously 0.001
    parser.add_argument('--wd', type=float, default=1e-2) # weight decay. Other options: 1e-5 or 1e-7 - previously 1e-4

    return parser.parse_args()
