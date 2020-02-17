import ij.*;
import ij.process.*;
import ij.gui.*;
import java.awt.*;
import ij.plugin.*;
import ij.plugin.frame.*;
import java.lang.Math;
public class ex_03_2_jpfeif2sGausFilter implements PlugIn {
//input is a grayscale image
//add SaltAndPepper noise and try to remove it with a gaussian filter 5x5, 7x7
	
	private int gausCount=1;
	public void run(String arg) {
		ImagePlus imp = IJ.getImage();
		//ImagePlus noisyImage = addSaltAndPepper(imp.duplicate());
		//IJ.save(noisyImage, noisyImage.getTitle());
		
		//use gausfilter 5x5
		ImagePlus gaus5 = gaussianFilter(imp.duplicate(),5,1);
		for(int i = 0;i<2;i++) {
			gaus5 = gaussianFilter(gaus5, 5, 1);
		}
        
		//gausCount = 1;
		//use gausfilter 7x7
		//ImagePlus gaus7 = gaussianFilter(noisyImage.duplicate(),7,1.2);
		//for(int i = 0;i<2;i++) {
		//	gaus7 = gaussianFilter(gaus7, 7, 1.2);
		//}
		//s = m2-1 mit 2o<m<3m
		
	}
	private ImagePlus addSaltAndPepper (ImagePlus imp){
		//add salt and pepper noise to image
		ImageProcessor ip = imp.getProcessor();
		imp.setTitle("ex_03_2_SaltAndPepperNoise");
		int w = ip.getWidth();
		int h = ip.getHeight();

		int[] c = new int[1]; 
		int[] salt = {255};
		int[] pepper = {0};
		for(int u = 0; u<w;u++)
		{
			for(int v = 0; v<h;v++)
			 {
			    // random determines if the pixel will be salt, pepper or no noise
				//salt at rand < 0.02
				//pepper at rand > 0.98
				// no noise at  [0.02-0.98]
				double rand = Math.random();
				if(rand < 0.02) {
					ip.putPixel(u,v,salt);
				} else if (rand > 0.98) {
					ip.putPixel(u,v,pepper);
				}
				
			}
		}
		imp.show();
		imp.draw();
		return imp;
	}
	private ImagePlus gaussianFilter(ImagePlus imp, int size, double o) {
		ImageProcessor ip = imp.getProcessor();
        
		int w = ip.getWidth();
		int h = ip.getHeight();
		ImagePlus resultImage = IJ.createImage("ex_03_2_GaussianFilter"+size+"_Anwendung"+gausCount, "8-Bit",w,h,1);
		gausCount++;
		ImageProcessor resultIp = resultImage.getProcessor();
		
		int[] c = new int[1]; 
		for(int u = 0; u<w;u++)
		{
			for(int v = 0; v<h;v++)
			 {
				
				//create the mask
				double newVal = 0.0;
				for(int lineCounter = -(size/2); lineCounter <=(size/2); lineCounter++) {
					for(int rowCounter = -(size/2); rowCounter <=(size/2); rowCounter++) {
						c = ip.getPixel((u+rowCounter)%w,(v+lineCounter)%h,c);
						newVal += gausWeight(rowCounter,lineCounter,o)*(double)c[0];
					}
				}
				
				resultIp.putPixel(u,v,(int)newVal);
			 }
		}
		resultImage.show();
		resultImage.draw();
		return resultImage;
	}
	
	private double gausWeight(double u,double v,double o) {
		return ((1/(2*Math.PI*(Math.pow(o,2))))*(Math.pow(Math.E,-((Math.pow(u,2)+Math.pow(v,2))/(2*Math.pow(o,2))))));
	}

}
