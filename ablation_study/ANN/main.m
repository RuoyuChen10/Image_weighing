clc;
clear;

input = []
for i = 1:50
    img = imread(strcat("./fit_images", [num2str(i),'.jpg']))
    img(find(img>30))=255;
    img(find(img<30))=0;

    %�����
    round_area = regionprops(img,'Area');
    fprintf('round_area = %f\n', round_area.Area);
    area=cat(1,round_area.Area);
    area=area(255);

    %���ܳ�
    girth = regionprops(img,'Perimeter');
    fprintf('s.Perimeter = %f\n', girth.Perimeter);
    zhouc=cat(1,girth.Perimeter);
    zhouc=zhouc(255);

    %'ConvexArea'
    convex_area =  regionprops(img,'ConvexArea');
    fprintf('s.ConvexArea = %f\n', convex_area.ConvexArea);
    convex_area=cat(1,convex_area.ConvexArea);
    convex_area=convex_area(255);

    %'MajorAxisLength'
    MajorAxisLength = regionprops(img,'MajorAxisLength');
    fprintf('s.MajorAxisLength = %f\n', MajorAxisLength.MajorAxisLength);
    MajorAxisLength=cat(1,MajorAxisLength.MajorAxisLength);
    MajorAxisLength=MajorAxisLength(255);

    %MinorAxisLength
    MinorAxisLength = regionprops(img,'MinorAxisLength');
    fprintf('s.MinorAxisLength = %f\n', MinorAxisLength.MinorAxisLength);
    MinorAxisLength=cat(1,MinorAxisLength.MinorAxisLength);
    MinorAxisLength=MinorAxisLength(255);
    input(i,:)=[area,zhouc,convex_area,MajorAxisLength,MinorAxisLength]
end
xlswrite('result.xls',input)
weight=textread('weight.txt')
weight=weight(:,1)'
input=input'