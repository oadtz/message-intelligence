FROM php:7.1-fpm
RUN apt-get update \
    && apt-get install htop
RUN pecl install redis \
    && docker-php-ext-enable redis