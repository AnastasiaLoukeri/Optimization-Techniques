function [a_hist, b_hist, k_hist] = bisection_search_history(f, a, b, epsilon, limit)

    ak = a;
    bk = b;
    k = 0; % Δείκτης επαναλήψεων

    % Αρχικοποίηση καταγραφής ιστορικού
    a_hist = ak;
    b_hist = bk;
    k_hist = k;
    
    if epsilon <= 0 || (2*epsilon >= limit)
        % Μη έγκυρες παράμετροι
        return;
    end

    while (bk - ak) > limit
        k = k + 2;
        
        % Υπολογισμός σημείων διαχωρισμού
        pix1 = (ak + bk) / 2 - epsilon;
        pix2 = (ak + bk) / 2 + epsilon;

        % Υπολογισμοί συνάρτησης
        fpix1 = f(pix1); 
        fpix2 = f(pix2);
        
        % Μείωση διαστήματος
        if fpix1 < fpix2 
            bk = pix2; 
        elseif fpix1 > fpix2 
            ak = pix1; 
        else 
            ak = pix1;
            bk = pix2;
        end
        
        % Αποθήκευση ιστορικού για την τρέχουσα επανάληψη
        a_hist(end+1) = ak; 
        b_hist(end+1) = bk;
        k_hist(end+1) = k;
    end
end