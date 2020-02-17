import ij.*;
import ij.process.*;
import ij.gui.*;
import java.awt.*;
import ij.plugin.*;
import ij.plugin.frame.*;
import java.lang.Math;
public class ex_04_1_jpfeif2sAdaptiveThreshhold implements PlugIn {
//input is a grayscale image
//output is binary grayscale image
//using local adaptive threshhold

	public void run(String arg) {
		ImagePlus imp = IJ.getImage();
		ImagePlus resImage = localAdaptiveThreshhold(imp.duplicate(),3000);
		//IJ.save(resImage, resImage.getTitle());
		//resImage = localAdaptiveThreshhold(imp.duplicate(),100);
		//IJ.save(resImage, resImage.getTitle());
	}

	private ImagePlus localAdaptiveThreshhold(ImagePlus imp,int size) {
		//transform into a binary image
		ImageProcessor ip = imp.getProcessor();
		int w = ip.getWidth();
		int h = ip.getHeight();
		ImagePlus resultImage = IJ.createImage("ex_04_1_Threshold_"+size+".png", "8-Bit",w,h,1);
		ImageProcessor resultIp = resultImage.getProcessor();

		int[] c = new int[3]; 
		int newVal=0;
		//create box
		for(int lineCounter = 0; lineCounter <h; lineCounter+=size) {
			for(int rowCounter = 0; rowCounter <w; rowCounter+=size) {//for each box
				//find best threshhold
				double max=0;
				double threshhold=-1;
				double m0=0;
				double m1=0;
				double m=0;
				double o0 = 0;
				double o1 =0;
				double ozw=0;
				double oin=0;
				double p0=0;
				double p1=0;
				double count0=0;
				double count1=0;
				//get probabilities of the individual graysale values
				double[] p = localProbabilityOfValues(lineCounter,rowCounter,size,ip);
				for(int t = 0;t<256;t++) { //for one possible threshold
					m0=0;
					m1=0;
					count0 =0;
					count1 =0;
					o0=0;
					o1=0;
					p1=0;
					p0=0;
					for(int u = lineCounter; (u<w && u<lineCounter+size);u++)//get m0 and m1
					{
						for(int v = rowCounter; (v<h&&v<rowCounter+size);v++){
							//determine mean0 and mean1. mean for below and above the threshold
							c = ip.getPixel(u,v,c);
							
							if(c[0]<=t) {
								m0+=c[0]; //is black
								count0++;
							}else {
								m1+=c[0];
								count1++;
							}
						}
					}
					m0/=count0;
					m1/=count1;
					
					p0 = p0(t, p);
					p1 = 1-p0;
					m = m0*p0 + m1*p1;
					for(int g=0;g<=t;g++) {
						o0+= Math.pow((double)g-m0,2)*p[g];
					}
					for (int g=t+1;g<=256-1;g++) {
						o1+= Math.pow((double)g-m1,2)*p[g];
					}
					oin = p0*o0+p1*o1;
					ozw = p0*Math.pow(m0-m,2)+p1*Math.pow(m1-m,2);
					if((ozw/oin)>max) {
						max = ozw/oin;
						threshhold = t;
					}
					//t is the current threshold attempt
					
					//IJ.log("(line="+lineCounter+" row="+rowCounter+")   best threshhold="+threshhold+" MAX="+max+"  vs  RATIO="+(ozw/oin)+" TRIED: T="+t+" (p0="+p0+" p1="+p1+" oin="+oin+" ozw="+ozw);
				}//end threshold finiding process
				//threshhold is now the best threshhold for this part of the picture
				
				//go thorugh the pixels of one box
				for(int u = lineCounter; (u<w && u<lineCounter+size);u++)
				{
					for(int v = rowCounter; (v<h&&v<rowCounter+size);v++){
						//make the class binary according to the threshold t
						c = ip.getPixel(u,v,c);
						if(c[0]<=threshhold) {
							newVal = 0; //black
						}else {
							newVal = 255; //white
						}
						resultIp.putPixel(u,v,newVal);
					}
				}
			}
		}
		resultImage.show();
		resultImage.draw();
		return resultImage;
	}
	//returns a double array with the local probabilites p(g) for each possible grayscale value.
	private double[] localProbabilityOfValues(int lineCounter, int rowCounter, int size, ImageProcessor ip) {
		int w = ip.getWidth();
		int h = ip.getHeight();
		double[] prob = new double[256];
		int[] c = new int[3]; 
		double count=0;
		//determine how often each color is present in the image
		for(int i =0; i<prob.length;i++) {
			prob[i] = 0.0;
		}
		for(int u = lineCounter; ((u<w) && (u<(lineCounter+size)));u++)
		{
			for(int v = rowCounter; ((v<h) && (v<(rowCounter+size)));v++){
				c = ip.getPixel(u,v,c);
				prob[c[0]]++;
				count ++;
			 }
		}
		//divide the absolute occurrence by the number of pixels
		for(int i=0;i<prob.length;i++) {
			prob[i]= prob[i]/(count);
		}
		
		return prob;
	}
	
	//returns the probability P0(t)= sum (g=0, t, (p(g))
	private double p0(int t, double[] p) {
		double ret =0;
		for(int g = 0; g<=t;g++) {
			ret += p[g];
		}
		return ret;
	}
	

}
