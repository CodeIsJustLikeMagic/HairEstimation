import ij.*;
import ij.process.*;
import ij.gui.*;
import java.awt.*;
import ij.plugin.*;
import ij.plugin.frame.*;

public class ex_02_2_jpfeif2sGrayscaleRescaleUp implements PlugIn {
//grayscale [100,200] to grayscale[0,255] scale up
	public void run(String arg) {
		ImagePlus imp = IJ.getImage();
		grayscale_0_255(imp.duplicate());
		
	}
	private void grayscale_0_255 (ImagePlus imp){
		ImageProcessor ip = imp.getProcessor();

		
		int w = ip.getWidth();
		int h = ip.getHeight();
		
		//create blank 8-bit grayscale images in the correct size
		ImagePlus grayImage = IJ.createImage("grayscale 0_255", "8-bit grayscale",w,h,1);
		ImageProcessor gIp = grayImage.getProcessor();

		int[] c = new int[3]; 
		//go through all the pixels
		for(int u = 0; u<w;u++)
		{
			for(int v = 0; v<h;v++)
			 {
				//get colors
				c = ip.getPixel(u,v,c);

				//input is a grayscale image.
				//adjusting values to b_low = 0, b_high=255
				int g = c[0];
				g = (int) ((g-100)*(255.0/100.0));
				gIp.putPixel(u,v,g);
				
			}
		}
		//display images	
		grayImage.show();
		grayImage.draw();
		}
}
