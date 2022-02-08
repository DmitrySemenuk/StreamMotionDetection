//
//  OpenCVWrapper.h
//  StreamMotionDetection
//
//  Created by Дмитрий Семенюк on 8.02.22.
//

#import <UIKit/UIKit.h>
#import <AVFoundation/AVFoundation.h>

@interface OpenCVWrapper : NSObject

+ (UIImage *)backgroundSubstract:(UIImage *)image;

@end
