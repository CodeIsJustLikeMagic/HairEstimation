import ij.*;
import ij.process.*;
import ij.gui.*;
import java.awt.*;
import ij.plugin.*;
import ij.plugin.frame.*;
import java.lang.Math;
public class ex_03_1_jpfeif2sCrossCorrelationFilter implements PlugIn {
//cross correlation filter, size 3x3 pixels, difference of color channel to lumiance on the middle pixel
	public void run(String arg) {
		ImagePlus imp = IJ.getImage();
		crossCorrelationFilter(imp.duplicate());
		
		
	}
	private void crossCorrelationFilter (ImagePlus imp){
		ImageProcessor ip = imp.getProcessor();

		int w = ip.getWidth();
		int h = ip.getHeight();
		
		//create blank 8-bit grayscale images in the correct size
		ImagePlus resultImage = IJ.createImage("ex_03_1_Nachher (KreuzkorrelationsFilter)", "8-Bit",w,h,1);
		ImageProcessor resultIp = resultImage.getProcessor();

		int[] c = new int[3]; 
		int[][] mask = new int[9][3];
		for(int u = 0; u<w;u++)
		{
			for(int v = 0; v<h;v++)
			 {
				
				//get colors
				//get the 9 pixels in the mask. use % to warp around at the edges
				//first line
				mask[0] = ip.getPixel((u-1)%w,(v-1)%h,c);
				mask[1] = ip.getPixel((u)%w,(v-1)%h,c);
				mask[2] = ip.getPixel((u+1)%w,(v-1)%h,c);
				//second line
				mask[3] = ip.getPixel((u-1)%w,(v)%h,c);
				mask[4] = ip.getPixel((u)%w,(v)%h,c);
				mask[5] = ip.getPixel((u+1)%w,(v)%h,c);
				//third line
				mask[6] = ip.getPixel((u-1)%w,(v+1)%h,c);
				mask[7] = ip.getPixel((u)%w,(v+1)%h,c);
				mask[8] = ip.getPixel((u+1)%w,(v+1)%h,c);
				
				//calculate luminance for the middle pixel
				c = ip.getPixel (u,v,c);
				int lum =(int) ((0.3*(double)c[0])+(0.6*(double)c[1])+(0.1*(double)c[2]));
				
				//find largest difference |r-lum|, |g-lum|, |b-lum|
				int maxDif = 0;
				for(int pixel = 0; pixel<mask.length; pixel++) {
					for(int channel =0; channel<3;channel++) {
						int dif = Math.abs( mask[pixel][channel] - lum);
						if(dif > maxDif) {
							maxDif = dif; 
						}
					}
				}

				
				resultIp.putPixel(u,v,maxDif);
				
			}
		}
		//display images	
		resultImage.show();
		resultImage.draw();
		}

}
