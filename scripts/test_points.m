function test_points_distribution
        global N; N = 6^2 * 4; # N=k * k * 4, k - some integer
        plot_points();
end

function gm = gamma1(theta, phi)
        gm = [sin(theta)*cos(phi); sin(theta)*sin(phi); cos(theta)];
end

function gm = gamma2(theta, phi)
        gm = 0.5 * [sin(theta)*cos(phi); sin(theta)*sin(phi); cos(theta)];
end


function src_pnts = get_source_points()
        global N;

        count = sqrt(N / 4);
        ii = 1;
        for theta = linspace(0, pi, count)
                for phi = linspace(0, 2 * pi, 2 * count)
                        src_pnts(:, ii) = 2 * gamma1(theta, phi);
                        src_pnts(:, ii + N / 2) = 0.8 * gamma2(theta, phi);
                        ii += 1;
                end
        end
end

function [coll_pnts] = get_collocation_points(gamma, nu)
        global N;
        count = sqrt(N / 4); ii = 1;
        for theta = linspace(0, pi, count)
                for phi = linspace(0, 2 * pi, 2 * count)
                        coll_pnts(:, ii) = gamma(theta, phi);
                        ii += 1;
                end
        end
end

function plot_points()
##        yy = get_source_points();
##        scatter3(yy(1, :), yy(2, :), yy(3, :), 64, 'filled');
##        hold on;
        xx = get_collocation_points(@gamma1);
        scatter3(xx(1, :), xx(2, :), xx(3, :), 64, 'filled');
end


##[x, y, z] = sphere (40);
##surf (3*x, 3*y, 3*z);
##axis equal;
##title ("sphere of radius 3");


