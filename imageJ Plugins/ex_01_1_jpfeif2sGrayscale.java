import ij.*;
import ij.process.*;
import ij.gui.*;
import java.awt.*;
import ij.plugin.*;
import ij.plugin.frame.*;

public class ex_01_1_jpfeif2sGrayscale implements PlugIn {

	public void run(String arg) {
		ImagePlus imp = IJ.getImage();
		grayscale(imp.duplicate());
		
		
	}
	private void grayscale (ImagePlus imp){
		ImageProcessor ip = imp.getProcessor();

		
		int w = ip.getWidth();
		int h = ip.getHeight();
		
		//create blank 8-bit grayscale images in the correct size
		ImagePlus equalImage = IJ.createImage("equal input from colors", "8-bit grayscale",w,h,1);
		ImageProcessor equalIp = equalImage.getProcessor();
		ImagePlus lumImage = IJ.createImage("luminanz grayscale", "8-bit grayscale",w,h,1);
		ImageProcessor lumIp = lumImage.getProcessor();

		int[] c = new int[3]; 
		//go through all the pixels
		for(int u = 0; u<w;u++)
		{
			for(int v = 0; v<h;v++)
			 {
				//get colors
				c = ip.getPixel(u,v,c);

				//grayscale with equal parts from all colors
				int equalval = (int) ( ((double) (c[0]+c[1]+c[2])) / 3);
				int[] equ = {(int)equalval};
				equalIp.putPixel(u,v,equ);
				
				//grayscale with luminanz
				int lumval =(int) ((0.3*(double)c[0])+(0.6*(double)c[1])+(0.1*(double)c[2]));
				int[] lum = {(int)lumval};
				lumIp.putPixel(u,v,lum);
			}
		}
		//display images	
		equalImage.show();
		equalImage.draw();
		lumImage.show();
		lumImage.draw();
	}
}
